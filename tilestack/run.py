"""The internal registration of a block (python -m tilestack BLOCK_JSON OUT_DIR): the tile layout, the correspondences of
every pair of slabs up to data.NEIGHBOURS apart, the placement of all slabs and tiles, the transforms and the registered
volume with its source map, and with an MRI the affine of the volume to it (mri.py)."""
import glob
import hashlib
import json
import os
import pickle
import time

import numpy as np
import torch

from . import compose, data, export, match, solve, tiles


def settings():
    """What the caches in data.WORK depend on."""
    return dict(block={k: v for k, v in data.BLOCK.items() if k != "work"}, surface_tol=data.SURFACE_TOL, level_mm=data.LEVEL_MM,
                match=dict(block=match.BLOCK, coarse=match.COARSE, depths=match.DEPTHS, search=match.SEARCH, peak=match.PEAK,
                           min_blocks=match.MIN_BLOCKS, agree=match.AGREE, triples=match.TRIPLES, tilt=match.TILT))


def correspondences():
    """The good matches of every pair of slabs up to data.NEIGHBOURS apart (cached), and the pairs left out (match.NoOverlap,
    or no good match)."""
    os.makedirs(f"{data.WORK}/corr", exist_ok=True)
    m, L, skipped = [], {}, []
    for k in range(1, data.NEIGHBOURS + 1):
        for s, t in zip(data.SLABS, data.SLABS[k:]):
            path = f"{data.WORK}/corr/m64_{s:03d}_{t:03d}.npz"
            if not os.path.exists(path):
                for q in [q for q in L if q not in (s, t)]:
                    del L[q]
                for q in (s, t):
                    if q not in L:
                        L[q] = torch.from_numpy(data.load_log(q)).to(match.dev)
            try:
                c = match.match_pair(s, t, L.get(s), L.get(t), out=path)
            except match.NoOverlap as e:
                skipped.append(str(e))
                continue
            c = match.keep(c, match.good(c))
            if len(c["zB"]):
                m.append(c)
            else:
                skipped.append(f"pair {s}-{t}: no dense match kept")
    del L
    torch.cuda.empty_cache()
    joined = {data.SLABS[0]}
    while True:
        new = {int(c[q]) for c in m for q in ("s", "t") if {int(c["s"]), int(c["t"])} & joined} - joined
        if not new:
            break
        joined |= new
    if joined != set(data.SLABS):
        raise RuntimeError(f"the overlaps do not join slabs {sorted(set(data.SLABS) - joined)} to the rest: {skipped}")
    return m, skipped


def run(out, full=False, layout=None, mri=None):
    if mri:
        from .mri import octreg_params, register_mri
        octreg_params(mri)                                     # a wrong MRI file stops the run here, not at the end
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    data.check_work(settings())
    lay = tiles.layout(json.load(open(layout)) if layout else None)
    m, skipped = correspondences()
    sol = solve.solve(m, lay, data.SLABS)
    pickle.dump(sol, open(f"{out}/solution.pkl", "wb"))
    export.save_transforms(sol, lay, f"{out}/transforms.json")
    vol, vz0, src = compose.compose(sol, lay)
    export.save_volume(vol, vz0, data.LEVEL_MM * 1000, f"{out}/registered_40um.nii.gz", src)
    del vol, src
    if full:
        raw = f"registered_{data.VOX * 1000:g}um"
        vol, vz0, src = compose.compose(sol, lay, full=True, out_path=f"{out}/{raw}_tmp.npy")
        export.save_volume(vol, vz0, data.VOX * 1000, f"{out}/{raw}.nii", src)
        del vol, src
        os.remove(f"{out}/{raw}_tmp.npy")
    res = {}
    for k in range(1, data.NEIGHBOURS + 1):
        r = [np.abs(np.stack(solve.residuals(sol, lay, c))) for c in m if data.SLABS.index(int(c["t"])) - data.SLABS.index(int(c["s"])) == k]
        if r:
            r = np.concatenate(r, 1) * data.VOX * 1000
            xy = np.hypot(r[0], r[1])
            res[f"neighbours_{k}"] = dict(matches=int(r.shape[1]), inplane_um_median=float(np.median(xy)), inplane_um_p95=float(np.percentile(xy, 95)),
                                          depth_um_median=float(np.median(r[2])), depth_um_p95=float(np.percentile(r[2], 95)))
    code = hashlib.sha256(b"".join(open(p, "rb").read() for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))))).hexdigest()
    json.dump(dict(seconds=round(time.time() - t0), pairs=len(m), skipped=skipped, residuals=res, solver_iterations=sol["iterations"],
                   settings=sol["settings"], block=data.BLOCK, code=code[:12]),
              open(f"{out}/run.json", "w"), indent=1)
    if mri:
        register_mri(f"{out}/registered_40um.nii.gz", mri, f"{out}/mri")
    print("done", out, round(time.time() - t0), "s")
