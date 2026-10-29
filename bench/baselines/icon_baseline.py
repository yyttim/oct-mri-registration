#!/usr/bin/env python3
"""uniGradICON / multiGradICON on the I58 pair, reduced to an affine and put in the convention of the other baselines.

    python bench/baselines/icon_baseline.py run --inputs DIR --out DIR [--model multigradicon] [--io 50] [--starts all]
    python bench/baselines/icon_baseline.py selftest

The network is a deformable model for roughly aligned volumes of one FOV, so it gets what the other tools get from their
starts: the OCT is resampled onto the MRI grid under a start pose, the 24 rotations of the cube about the aligned box centres
(`r00` is the header orientation with the centres aligned) and the header alignment itself (`header`), each with the OCT
contrast as it is and inverted inside the FOV (`inv`). Both volumes go through the model's own 'mri' preprocessing and the
175^3 resize, with the instance optimisation (--io steps, LNCC) on top of the pretrained weights. The pose of a start is the
least-squares affine fit to the displacement field over the OCT footprint, composed with the start pose:
T_oct2mri = L^-1 T_start, with the fit y -> L y + t from MRI to start-image coordinates. The start with the lowest final
similarity loss of the model itself is the main run; no octreg quantity enters the choice. The residual of the affine fit
(rms of field minus fit, mm) says how far from affine the model's field is.

Needs torch, itk and unigradicon (the `venv_icon` environment); the writer of the run directories, `rundir.py`, runs in
the octreg environment: `python bench/baselines/rundir.py init RUN --T RUN/T_oct2mri.txt`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

D_LPS = np.diag([-1.0, -1.0, 1.0])


def cube_rotations():
    """The 24 proper rotations of the cube, as signed permutation matrices."""
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, p[i]] = s[i]
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def start_pose(R, c_oct, c_mri):
    """T_start: OCT world -> MRI world, rotation R about the OCT box centre onto the MRI box centre."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c_mri - R @ c_oct
    return T


def fit_affine(y, m):
    """Least squares m ~ L y + t for points y -> m [n, 3]. Returns the 4x4 matrix and the rms residual."""
    X = np.c_[y, np.ones(len(y))]
    sol, *_ = np.linalg.lstsq(X, m, rcond=None)
    H = np.eye(4)
    H[:3, :3] = sol[:3].T
    H[:3, 3] = sol[3]
    return H, float(np.sqrt(((X @ sol - m) ** 2).sum(1).mean()))


def pose_from_fit(H_fix2start, T_start):
    """T_oct2mri from the fit fixed(MRI) -> start-image coordinates and the start pose OCT -> MRI (start image = OCT through T_start)."""
    return np.linalg.inv(H_fix2start) @ T_start


def box_centre(affine, shape):
    ijk = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    return xyz.mean(0)


def load_nifti(path):
    """(array [i, j, k] float32, 4x4 RAS voxel affine) of a NIfTI file, read with itk."""
    import itk
    img = itk.imread(str(path), itk.F)
    sp = np.array(img.GetSpacing(), float)
    D = itk.array_from_matrix(img.GetDirection())
    A = np.eye(4)
    A[:3, :3] = D_LPS @ D * sp
    A[:3, 3] = D_LPS @ np.array(img.GetOrigin(), float)
    return np.ascontiguousarray(itk.array_from_image(img).transpose(2, 1, 0)), A


def itk_image(arr, affine):
    """A float32 itk image (LPS physical space) from an array on an RAS voxel affine."""
    import itk
    sp = np.linalg.norm(affine[:3, :3], axis=0)
    img = itk.image_from_array(np.ascontiguousarray(arr.transpose(2, 1, 0), dtype=np.float32))
    img.SetSpacing([float(s) for s in sp])
    img.SetOrigin([float(v) for v in D_LPS @ affine[:3, 3]])
    img.SetDirection(itk.matrix_from_array(D_LPS @ (affine[:3, :3] / sp)))
    return img


def resample_start(oct_arr, A_o, shape, A_m, T_start, device):
    """The OCT on the MRI grid under T_start (trilinear, zero outside), as a torch tensor [shape]."""
    import torch
    import torch.nn.functional as F
    X = torch.as_tensor(oct_arr, dtype=torch.float32, device=device)[None, None]
    inv = np.linalg.inv(A_o) @ np.linalg.inv(T_start) @ A_m              # MRI voxel -> OCT voxel
    o = np.array(oct_arr.shape, float)
    out = torch.zeros(shape, dtype=torch.float32, device=device)
    ii = torch.arange(shape[0], device=device, dtype=torch.float32)
    jj = torch.arange(shape[1], device=device, dtype=torch.float32)
    kk = torch.arange(shape[2], device=device, dtype=torch.float32)
    M = torch.as_tensor(inv, dtype=torch.float32, device=device)
    for i0 in range(0, shape[0], 32):
        i = ii[i0:i0 + 32]
        g = torch.stack(torch.meshgrid(i, jj, kk, indexing="ij"), -1)
        p = g @ M[:3, :3].T + M[:3, 3]
        n = 2 * p / torch.as_tensor(o - 1, dtype=torch.float32, device=device) - 1
        s = F.grid_sample(X, n.flip(-1)[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        inside = ((p >= 0) & (p <= torch.as_tensor(o - 1, dtype=torch.float32, device=device))).all(-1)
        out[i0:i0 + 32] = torch.where(inside, s, torch.zeros_like(s))
    return out


def run(a):
    import itk
    import torch
    import icon_registration.itk_wrapper as iw
    from unigradicon import get_multigradicon, get_unigradicon, make_sim, preprocess

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inp, out = Path(a.inputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mri, A_m = load_nifti(inp / "mri_0.08mm.nii.gz")
    oct_, A_o = load_nifti(inp / "oct_0.08mm.nii.gz")
    fov = oct_ > 0
    lo, hi = np.percentile(oct_[fov], [1, 99])
    oct_inv = np.where(fov, hi + lo - np.clip(oct_, lo, hi), 0).astype(np.float32)
    c_m, c_o = box_centre(A_m, mri.shape), box_centre(A_o, oct_.shape)
    rots = cube_rotations()
    starts = [("header", np.eye(4))] + [
        (f"r{k:02d}", start_pose(R, c_o, c_m)) for k, R in enumerate(rots)]
    if a.starts != "all":
        keep = a.starts.split(",")
        starts = [s for s in starts if s[0] in keep]
    fixed = preprocess(itk_image(mri, A_m), "mri")
    net_fn = {"unigradicon": get_unigradicon, "multigradicon": get_multigradicon}[a.model]
    net = net_fn(make_sim("lncc"))
    io_steps = None if a.io == 0 else a.io
    sub = (slice(None, None, 6),) * 3
    ci, cj, ck = [np.arange(n)[s] for n, s in zip(mri.shape, sub)]
    vox = np.stack(np.meshgrid(ci, cj, ck, indexing="ij"), -1).reshape(-1, 3).astype(float)
    pts_ras = vox @ A_m[:3, :3].T + A_m[:3, 3]
    pts_lps = pts_ras @ D_LPS.T

    rows = []
    for pol, src in (("raw", oct_), ("inv", oct_inv)):
        if a.polarity != "both" and a.polarity != pol:
            continue
        for name, T_s in starts:
            tag = f"{name}_{pol}"
            rd = out / tag
            if (rd / "result.json").exists() and not a.force:
                rows.append(json.loads((rd / "result.json").read_text()))
                continue
            t0 = time.time()
            img = resample_start(src, A_o, mri.shape, A_m, T_s, dev).cpu().numpy()
            if not np.ptp(img):
                print(tag, "empty overlap, skipped", flush=True)
                continue
            footprint = img[sub] > 0
            moving = preprocess(itk_image(img, A_m), "mri")
            phi_AB, _, loss = iw.register_pair(net, moving, fixed, finetune_steps=io_steps, return_artifacts=True)
            m_lps = np.array([list(phi_AB.TransformPoint([float(v) for v in p])) for p in pts_lps[footprint.ravel()]])
            m_ras = m_lps @ D_LPS.T
            H, rms = fit_affine(pts_ras[footprint.ravel()], m_ras)
            T = pose_from_fit(H, T_s)
            rd.mkdir(parents=True, exist_ok=True)
            np.savetxt(rd / "T_oct2mri.txt", T, fmt="%.10f")
            res = {"run": tag, "start": name, "polarity": pol, "model": a.model, "io_steps": a.io,
                   "similarity_loss": float(loss.similarity_loss), "all_loss": float(loss.all_loss),
                   "flips": float(loss.flips), "affine_fit_rms_mm": rms, "det": float(np.linalg.det(T[:3, :3])),
                   "T_oct2mri": T.tolist(), "seconds": round(time.time() - t0, 1)}
            (rd / "result.json").write_text(json.dumps(res, indent=1))
            rows.append(res)
            print(f"{tag}: sim {res['similarity_loss']:.4f} rms {rms:.2f} mm det {res['det']:.3f} {res['seconds']} s", flush=True)
    rows.sort(key=lambda r: r["similarity_loss"])
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print("best:", rows[0]["run"] if rows else None)


def selftest():
    rng = np.random.default_rng(0)
    A_o = np.diag([0.08, 0.08, 0.08, 1.0])
    A_m = np.eye(4)
    A_m[:3, :3] = np.diag([0.08] * 3)
    R = cube_rotations()[7]
    T_s = start_pose(R, box_centre(A_o, (50, 60, 70)), box_centre(A_m, (90, 80, 100)))
    assert len(cube_rotations()) == 24
    # a known truth: OCT -> MRI by a rotation and shift; the start image is OCT through T_start, so its coordinates are T_s o
    th = 0.3
    Rt = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    T_true = np.eye(4)
    T_true[:3, :3] = Rt * 1.02
    T_true[:3, 3] = [1.0, -2.0, 3.0]
    y = rng.uniform(0, 5, (400, 3))                       # MRI points
    o = (np.linalg.inv(T_true) @ np.c_[y, np.ones(400)].T).T[:, :3]
    m = (T_s @ np.c_[o, np.ones(400)].T).T[:, :3]         # where the model's field must send y: the start-image coordinates
    H, rms = fit_affine(y, m)
    assert rms < 1e-9 and np.allclose(pose_from_fit(H, T_s), T_true, atol=1e-9)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="multigradicon", choices=["unigradicon", "multigradicon"])
    r.add_argument("--io", type=int, default=50, help="instance-optimisation steps, 0 for the network alone")
    r.add_argument("--starts", default="all", help="comma list of header, r00..r23, or all")
    r.add_argument("--polarity", default="both", choices=["raw", "inv", "both"])
    r.add_argument("--force", action="store_true")
    sp.add_parser("selftest")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else selftest()
