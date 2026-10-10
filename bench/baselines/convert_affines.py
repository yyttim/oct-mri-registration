"""Transform-file conventions of the baseline tools -> one common object:
a 4x4 matrix T that maps MOVING-image world coordinates to FIXED-image world coordinates,
both in NIfTI RAS mm (nibabel's img.affine convention).

Sources for each convention (quoted in the function docstrings):
  ITK/ANTs  : itkMatrixOffsetTransformBase.hxx ComputeOffset(): offset = t + c - M c; GetFixedParameters() = center;
              ITK is LPS (Slicer coordinate-systems page); ANTs wiki "forward warps ... transform points from fixed to
              moving space"; lta_convert: "--initk ... ITK transform (inverse LPS2LPS)", inverse = target->source.
  elastix   : same ITK affine (TransformParameters = 9 matrix entries then translation; CenterOfRotationPoint in world mm).
  NiftyReg  : reg_aladin/reg_f3d usage "(Affine*Reference=Floating)"; lta_convert "--inniftyreg ... (inverse RAS2RAS)".
  FSL FLIRT : FLIRT FAQ: scaled-mm coordinates, x -> N-1-x when det(sform) > 0; fslpy fsl.transform.flirt:
              src voxels -> src scaled voxels -> ref scaled voxels (the FLIRT matrix) -> ref voxels -> ref world.
  FreeSurfer: mri_robust_register "--lta ... transform from mov to dst", "--vox2vox ... (default is RAS2RAS)";
              include/transform.h: LINEAR_VOX_TO_VOX 0, LINEAR_RAS_TO_RAS 1.
  greedy    : reference docs: matrix "maps voxels in fixed image space to voxels in moving image space" via ras().

Usage (CLI helpers):  python convert_affines.py scale-header in.nii.gz out.nii.gz 10
"""
import re
import sys

import numpy as np
import nibabel as nib

D_LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS <-> RAS, its own inverse


def _homog(M, t):
    A = np.eye(4)
    A[:3, :3] = M
    A[:3, 3] = t
    return A


# ----------------------------------------------------------------------------------------------- ITK / ANTs / elastix
def itk_affine_fixed2moving_lps(params, center):
    """ITK MatrixOffsetTransformBase: y = M x + offset with offset = t + c - M c  (ComputeOffset in the .hxx).
    params = 9 matrix entries row-major then 3 translations; center = FixedParameters.  Maps fixed -> moving, LPS mm."""
    p = np.asarray(params, float)
    M = p[:9].reshape(3, 3)
    t = p[9:12]
    c = np.asarray(center, float)
    return _homog(M, t + c - M @ c)


def lps_to_ras_matrix(A_lps):
    """Conjugate a point map by the LPS->RAS axis flip: A_ras = D A_lps D."""
    return D_LPS_RAS @ A_lps @ D_LPS_RAS


def ants_mat_to_mov2fix_ras(path):
    """ANTs *0GenericAffine.mat (ITK MATLAB-v4 transform file).  Returns T (moving world -> fixed world, RAS)."""
    from scipy.io import loadmat
    d = loadmat(path)
    key = [k for k in d if k.startswith("AffineTransform")][0]  # e.g. AffineTransform_double_3_3 / _float_3_3
    params = np.asarray(d[key], float).ravel()
    center = np.asarray(d["fixed"], float).ravel()
    A_fm_ras = lps_to_ras_matrix(itk_affine_fixed2moving_lps(params, center))
    return np.linalg.inv(A_fm_ras)


def _euler3d_matrix(rx, ry, rz, compute_zyx=False):
    """itk::Euler3DTransform::ComputeMatrix: Rz*Ry*Rx if ComputeZYX else Rz*Rx*Ry (ITK default ComputeZYX = false).
    Verify numerically with transformix (see verify_against_transformix) before trusting a rigid-stage parse."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx if compute_zyx else Rz @ Rx @ Ry


def _elastix_param(txt, name):
    m = re.search(r'\(%s\s+([^)]*)\)' % re.escape(name), txt)
    return m.group(1).replace('"', '').split() if m else None


def elastix_tp_fixed2moving_lps(path):
    """One TransformParameters.N.txt (AffineTransform or EulerTransform), following InitialTransformParameterFileName.
    elastix composes T(x) = T_last(... T_initial(x)), so the homogeneous product is A_last @ ... @ A_initial.
    Returns fixed -> moving in LPS mm (elastix works in ITK physical coordinates)."""
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
    elif kind == "TranslationTransform":
        A = _homog(np.eye(3), p[:3])
    else:
        raise ValueError("not a linear elastix transform: " + kind)
    init = _elastix_param(txt, "InitialTransformParameterFileName")
    if init and init[0] not in ("NoInitialTransform", ""):
        A = A @ elastix_tp_fixed2moving_lps(init[0])
    return A


def elastix_tp_to_mov2fix_ras(path):
    return np.linalg.inv(lps_to_ras_matrix(elastix_tp_fixed2moving_lps(path)))


def verify_against_transformix(A_fixed2moving_lps, outputpoints_txt, tol=1e-3):
    """transformix -def inputpoints.txt writes 'InputPoint = [ x y z ] ... OutputPoint = [ x y z ]' (LPS mm).
    Returns the max abs error between A @ InputPoint and OutputPoint."""
    err = 0.0
    for line in open(outputpoints_txt):
        ip = re.search(r'InputPoint = \[([^\]]*)\]', line)
        op = re.search(r'OutputPoint = \[([^\]]*)\]', line)
        if ip and op:
            x = np.r_[np.array(ip.group(1).split(), float), 1.0]
            y = np.array(op.group(1).split(), float)
            err = max(err, np.abs((A_fixed2moving_lps @ x)[:3] - y).max())
    return err


# ----------------------------------------------------------------------------------------------- NiftyReg / greedy
def niftyreg_txt_to_mov2fix_ras(path):
    """reg_aladin -aff text: 4x4, NIfTI world (RAS) mm, reference -> floating ("Affine*Reference=Floating")."""
    return np.linalg.inv(np.loadtxt(path))


def greedy_mat_to_mov2fix_ras(path):
    """greedy -a -o: 4x4, fixed -> moving in RAS physical coordinates (greedy reference docs)."""
    return np.linalg.inv(np.loadtxt(path))


# ----------------------------------------------------------------------------------------------- FSL FLIRT
def fsl_scaled_voxel_matrix(img):
    """voxel index -> FSL scaled-voxel (mm) coordinates: scale by pixdim; if det(voxel->mm) > 0 flip x -> N-1-x first
    (FLIRT FAQ 'What is the format of the matrix used by FLIRT'; fslpy 'fsl' coordinate system)."""
    pix = np.asarray(img.header.get_zooms()[:3], float)
    S = np.diag([pix[0], pix[1], pix[2], 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        F = np.eye(4)
        F[0, 0] = -1.0
        F[0, 3] = img.shape[0] - 1
        return S @ F
    return S


def flirt_mat_to_mov2fix_world(mat_path, src_img, ref_img):
    """FLIRT -omat maps src scaled-voxel -> ref scaled-voxel.  Chain (fslpy flirtMatrixToSform):
    src world -> src voxel -> src scaled -> ref scaled -> ref voxel -> ref world.  Cross-check with
    fsl.transform.flirt.fromFlirt(mat, src, ref, 'world', 'world')."""
    M = np.loadtxt(mat_path)
    SFs, SFr = fsl_scaled_voxel_matrix(src_img), fsl_scaled_voxel_matrix(ref_img)
    return ref_img.affine @ np.linalg.inv(SFr) @ M @ SFs @ np.linalg.inv(src_img.affine)


# ----------------------------------------------------------------------------------------------- FreeSurfer LTA
def lta_to_mov2fix_ras(path, src_img=None, dst_img=None):
    """LTA: 'type = 1' LINEAR_RAS_TO_RAS maps src (mov) RAS -> dst RAS directly.  'type = 0' VOX2VOX needs the two
    vox2ras matrices (pass the images; or first run lta_convert --inlta x.lta --outlta x_ras.lta, default RAS2RAS)."""
    txt = open(path).read()
    t = int(re.search(r'^\s*type\s*=\s*(\d+)', txt, re.M).group(1))
    m = re.search(r'^\s*1 4 4\s*\n((?:.*\n){4})', txt, re.M)
    M = np.array([[float(v) for v in l.split()] for l in m.group(1).strip().splitlines()])
    if t == 1:
        return M
    if t == 0:
        return dst_img.affine @ M @ np.linalg.inv(src_img.affine)
    raise ValueError("unsupported LTA type %d (convert with lta_convert --outlta)" % t)


# ----------------------------------------------------------------------------------------------- header scaling trick
def scale_header(img, s):
    """Multiply the voxel->world map by s about the world origin (A' = S A, S = diag(s,s,s,1)); voxel data unchanged.
    Used for tools whose internal scales are in mm (FLIRT 8/4/2/1 mm schedule, SynthMorph 1-mm 256-voxel space)."""
    S = np.diag([s, s, s, 1.0])
    A = S @ img.affine
    out = nib.Nifti1Image(np.asanyarray(img.dataobj), A, img.header)
    out.header.set_zooms(tuple(float(z) * s for z in img.header.get_zooms()[:3]))
    out.set_sform(A, code=1)
    out.set_qform(A, code=1)
    return out


def unscale_world_transform(T_scaled, s):
    """If both headers were scaled by S and T' maps scaled moving world -> scaled fixed world, then
    T = S^-1 T' S maps the original worlds (my derivation; check: T maps x to S^-1 T' S x)."""
    S = np.diag([s, s, s, 1.0])
    return np.linalg.inv(S) @ T_scaled @ S


# ----------------------------------------------------------------------------------------------- apply for QC
def resample_moving_to_fixed(moving_img, fixed_img, T_mov2fix_world, order=1, slab=32):
    """Resample moving onto the fixed grid with T (moving world -> fixed world); linear interpolation, 0 outside."""
    from scipy.ndimage import map_coordinates
    mov = np.asanyarray(moving_img.dataobj).astype(np.float32)
    fix2movvox = np.linalg.inv(moving_img.affine) @ np.linalg.inv(T_mov2fix_world) @ fixed_img.affine
    nx, ny, nz = fixed_img.shape[:3]
    out = np.zeros((nx, ny, nz), np.float32)
    jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    for i0 in range(0, nx, slab):
        ii = np.arange(i0, min(nx, i0 + slab))
        I, J, K = np.meshgrid(ii, np.arange(ny), np.arange(nz), indexing="ij")
        vox = np.stack([I.ravel(), J.ravel(), K.ravel(), np.ones(I.size)], 0).astype(np.float64)
        mv = fix2movvox @ vox
        out[ii[0]:ii[-1] + 1] = map_coordinates(mov, mv[:3], order=order, cval=0.0, mode="constant").reshape(I.shape)
    return nib.Nifti1Image(out, fixed_img.affine)


def apply_convexadam_displacement(moving_on_fixed_grid, disp_vox, order=1):
    """convex_adam_pt returns (H,W,D,3) displacements in voxels of the shared grid (fixed-grid, pull/backward warp)."""
    from scipy.ndimage import map_coordinates
    mov = np.asanyarray(moving_on_fixed_grid.dataobj).astype(np.float32)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in mov.shape], indexing="ij"), 0).astype(np.float32)
    coords = grid + np.moveaxis(disp_vox.astype(np.float32), -1, 0)
    out = map_coordinates(mov, coords, order=order, cval=0.0, mode="constant")
    return nib.Nifti1Image(out, moving_on_fixed_grid.affine)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "scale-header":
        nib.save(scale_header(nib.load(sys.argv[2]), float(sys.argv[4])), sys.argv[3])
    else:
        print(__doc__)
