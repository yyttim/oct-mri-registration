"""Transform files of the baseline tools as one 4x4 matrix T: moving world -> fixed world, NIfTI RAS mm.
The conventions follow ITK ComputeOffset, the ANTs wiki, the reg_aladin usage text, the FLIRT FAQ with fslpy, FreeSurfer
include/transform.h and the greedy reference docs.

    python bench/baselines/convert_affines.py scale-header in.nii.gz out.nii.gz 10
"""
import re
import sys

import numpy as np
import nibabel as nib

D_LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])  # its own inverse


def _homog(M, t):
    A = np.eye(4)
    A[:3, :3] = M
    A[:3, 3] = t
    return A


def itk_affine_fixed2moving_lps(params, center):
    """An ITK affine (MatrixOffsetTransformBase) as a 4x4 map fixed -> moving in LPS mm.
    params: 9 matrix entries row-major then 3 translations. center: the FixedParameters."""
    p = np.asarray(params, float)
    M = p[:9].reshape(3, 3)
    t = p[9:12]
    c = np.asarray(center, float)
    return _homog(M, t + c - M @ c)


def lps_to_ras_matrix(A_lps):
    return D_LPS_RAS @ A_lps @ D_LPS_RAS


def ants_mat_to_mov2fix_ras(path):
    """T from an ANTs *0GenericAffine.mat, an ITK transform in MATLAB format."""
    from scipy.io import loadmat
    d = loadmat(path)
    key = [k for k in d if k.startswith("AffineTransform")][0]  # AffineTransform_double_3_3 or _float_3_3
    params = np.asarray(d[key], float).ravel()
    center = np.asarray(d["fixed"], float).ravel()
    A_fm_ras = lps_to_ras_matrix(itk_affine_fixed2moving_lps(params, center))
    return np.linalg.inv(A_fm_ras)


def _euler3d_matrix(rx, ry, rz, compute_zyx=False):
    """The rotation of itk::Euler3DTransform, Rz Rx Ry by default."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx if compute_zyx else Rz @ Rx @ Ry


def _elastix_param(txt, name):
    m = re.search(r'\(%s\s+([^)]*)\)' % re.escape(name), txt)
    return m.group(1).replace('"', '').split() if m else None


def elastix_tp_fixed2moving_lps(path):
    """An affine or Euler elastix TransformParameters file with its initial transforms, as a map fixed -> moving in LPS mm."""
    txt = open(path).read()
    kind = _elastix_param(txt, "Transform")[0]
    p = np.array(_elastix_param(txt, "TransformParameters"), float)
    c = np.array(_elastix_param(txt, "CenterOfRotationPoint"), float)
    if kind == "AffineTransform":
        A = itk_affine_fixed2moving_lps(p, c)
    elif kind == "EulerTransform":
        zyx = (_elastix_param(txt, "ComputeZYX") or ["false"])[0].lower() == "true"
        M = _euler3d_matrix(*p[:3], compute_zyx=zyx)
        A = itk_affine_fixed2moving_lps(np.r_[M.ravel(), p[3:6]], c)
    else:
        raise ValueError("not an affine or Euler elastix transform: " + kind)
    init = _elastix_param(txt, "InitialTransformParameterFileName")
    if init and init[0] not in ("NoInitialTransform", ""):
        A = A @ elastix_tp_fixed2moving_lps(init[0])  # elastix applies the initial transform first
    return A


def elastix_tp_to_mov2fix_ras(path):
    return np.linalg.inv(lps_to_ras_matrix(elastix_tp_fixed2moving_lps(path)))


def niftyreg_txt_to_mov2fix_ras(path):
    """T from a reg_aladin -aff file, which maps reference to floating in RAS mm."""
    return np.linalg.inv(np.loadtxt(path))


def greedy_mat_to_mov2fix_ras(path):
    """T from a greedy -a -o file, which maps fixed to moving in RAS mm."""
    return np.linalg.inv(np.loadtxt(path))


def fsl_scaled_voxel_matrix(img):
    """Voxel index -> FSL scaled-voxel coordinates in mm, as in the FLIRT FAQ."""
    pix = np.asarray(img.header.get_zooms()[:3], float)
    S = np.diag([pix[0], pix[1], pix[2], 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        F = np.eye(4)
        F[0, 0] = -1.0
        F[0, 3] = img.shape[0] - 1
        return S @ F
    return S


def flirt_mat_to_mov2fix_world(mat_path, src_img, ref_img):
    """T from a FLIRT -omat matrix, which maps source scaled voxels to reference scaled voxels."""
    M = np.loadtxt(mat_path)
    SFs, SFr = fsl_scaled_voxel_matrix(src_img), fsl_scaled_voxel_matrix(ref_img)
    return ref_img.affine @ np.linalg.inv(SFr) @ M @ SFs @ np.linalg.inv(src_img.affine)


def lta_to_mov2fix_ras(path):
    """T from an LTA of type 1, LINEAR_RAS_TO_RAS, which maps src to dst."""
    txt = open(path).read()
    t = int(re.search(r'^\s*type\s*=\s*(\d+)', txt, re.M).group(1))
    if t != 1:
        raise ValueError(f"{path}: LTA type {t}, not RAS2RAS (lta_convert --inlta {path} --outlta OUT.lta converts it)")
    m = re.search(r'^\s*1 4 4\s*\n((?:.*\n){4})', txt, re.M)
    return np.array([[float(v) for v in l.split()] for l in m.group(1).strip().splitlines()])


def scale_header(img, s):
    """The image with its voxel -> world map scaled by s about the world origin, voxel data unchanged.
    For tools whose schedule is in mm, as FLIRT's."""
    S = np.diag([s, s, s, 1.0])
    A = S @ img.affine
    out = nib.Nifti1Image(np.asanyarray(img.dataobj), A, img.header)
    out.header.set_zooms(tuple(float(z) * s for z in img.header.get_zooms()[:3]))
    out.set_sform(A, code=1)
    out.set_qform(A, code=1)
    return out


def unscale_world_transform(T_scaled, s):
    """The transform between the original worlds from one between headers both scaled by s."""
    S = np.diag([s, s, s, 1.0])
    return np.linalg.inv(S) @ T_scaled @ S


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "scale-header":
        nib.save(scale_header(nib.load(sys.argv[2]), float(sys.argv[4])), sys.argv[3])
    else:
        print(__doc__)
