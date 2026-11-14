"""Bonus: fix the camera.

The camera of `loop` and `lost_at_sea` has radial lens distortion:  p_d = c + (p - c) (1 + kappa |p - c|^2 / 600^2),
where p is where an ideal pinhole camera would put a point and p_d where it appears in the frame.

    python bonus_camera.py                    all three synthetic flights (kappa is estimated for each)
    python bonus_camera.py loop lost_at_sea   some of them
    python bonus_camera.py --sift             use OpenCV's SIFT instead of your Part 1 features

What it does, for every flight:
  1. Estimates kappa from its own matches (no ground truth): for a candidate kappa it undistorts the matched points of
     pairs 5 frames apart (a bigger baseline shows the distortion better) of frames and measures how well ONE homography explains (RANSAC, truncated squared error). Between
     consecutive frames the ground is a plane, so the right kappa is the one with the smallest error.
     A coarse grid, then two finer grids around the best value.
  2. Undistorts the frames, finds the features again (dropping those near the empty border the undistortion creates).
  3. Repeats Q3.2 (chain with the similarity and with the homography) and the match-error plot.
The ground truth is used only to score the result, never to find kappa.

Writes to results/bonus/:  bonus_kappa.md, bonus_models.md, bonus_paths_<flight>.png,
                           bonus_error_vs_radius_<flight>.png, bonus_undistort_<flight>.jpg, T_<flight>_{sim,hom}_undistorted.npy
"""
import os, sys
import numpy as np, cv2
import utils, gt, viz
from utils import apply
from features import features, features_sift, match
from fitting import fit_homography, ransac
from odometry import chain

R0 = 600.0
CENTER = np.array([320.0, 240.0])


def distort(p, kappa, c=CENTER):
    """Ideal pixels -> pixels where they appear in the frame."""
    d = p - c
    return c + d * (1 + kappa * (d ** 2).sum(-1, keepdims=True) / R0 ** 2)


def undistort(pd, kappa, c=CENTER, iters=10):
    """Inverse of distort(): solve r_d = r (1 + kappa r^2 / 600^2) for r by Newton's method, point by point."""
    d = pd - c
    rd = np.linalg.norm(d, axis=-1, keepdims=True)
    r = rd.copy()
    for _ in range(iters):
        f = r * (1 + kappa * r ** 2 / R0 ** 2) - rd
        r = r - f / (1 + 3 * kappa * r ** 2 / R0 ** 2)
    scale = np.where(rd > 1e-9, r / np.maximum(rd, 1e-9), 1.0)
    return c + d * scale


def undistort_image(img, kappa):
    """Frame as an ideal camera would have taken it: output pixel p takes its value from distort(p)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src = distort(np.dstack([xx, yy]), kappa, np.array([w / 2, h / 2]))
    mx, my = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    valid = cv2.remap(np.ones((h, w), np.uint8) * 255, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    return out, valid


def score(pairs, kappa, tau=2.0):
    """How badly one homography per pair explains the matches after undistorting with kappa (lower is better).
    `pairs` holds matches that a loose RANSAC accepted. For each pair: undistort, fit a homography by least squares,
    refit on the matches within 2 tau of it, and sum min(error^2, tau^2) over all of them."""
    total = 0.0
    for p, q in pairs:
        pu, qu = undistort(p, kappa), undistort(q, kappa)
        # undistorting with a larger kappa shrinks the frame, and shrinks the errors with it: measure them in the
        # units of the original pixels, or the search would run off to the kappa that shrinks the most
        shrink = np.linalg.norm(pu - CENTER, axis=1).mean() / np.linalg.norm(p - CENTER, axis=1).mean()
        e = np.full(len(p), np.inf)
        keep = np.ones(len(p), bool)
        for _ in range(3):
            if keep.sum() < 8:
                break
            e = np.linalg.norm(apply(fit_homography(pu[keep], qu[keep]), pu) - qu, axis=1) / shrink
            keep = e < 2 * tau
        total += float(np.minimum(e ** 2, tau ** 2).sum())
    return total


def estimate_kappa(F, step=3, gap=5):
    """kappa that makes the matches of pairs 5 frames apart (a bigger baseline shows the distortion better) fit homographies best (grid search, three scales)."""
    pairs = []
    for i in range(0, len(F) - gap, step):
        (k1, d1), (k2, d2) = F[i], F[i + gap]
        mm = match(d1, d2)
        if len(mm) < 30:
            continue
        p, q = k1[mm[:, 0]], k2[mm[:, 1]]
        T, inl, _ = ransac(p, q, fit_homography, 4, 8.0, max_iter=500)       # loose: the points are still distorted
        if T is not None and inl.sum() >= 30:
            pairs.append((p[inl], q[inl]))
    best, half = 0.0, 0.4
    curve = None
    for stepk in (0.04, 0.008, 0.0016):
        grid = np.arange(best - half, best + half + 1e-9, stepk)
        s = [score(pairs, k) for k in grid]
        best = float(grid[int(np.argmin(s))])
        half = 2 * stepk
        curve = curve or (grid, s)
    return best, len(pairs)


def main():
    feat = features_sift if '--sift' in sys.argv else features
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or ['corniche', 'loop', 'lost_at_sea']
    out = utils.results_dir('bonus')
    if not os.path.isdir(out):
        os.makedirs(out)
    krows, mrows = [], []
    for fl in names:
        path = utils.flight(fl)
        print(f'... {fl}: features of the original frames', flush=True)
        imgs = utils.load_frames(path)
        F0 = [feat(im) for im in imgs]
        kappa, npairs = estimate_kappa(F0)
        print(f'    kappa = {kappa:.4f} (from {npairs} pairs)', flush=True)
        krows.append([fl, npairs, kappa])

        print(f'... {fl}: undistorting and recomputing features', flush=True)
        F1, valid_last = [], None
        for im in imgs:
            u, valid = undistort_image(im, kappa)
            valid = cv2.erode(valid.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0   # keep away from the border
            k, d = feat(u)
            if len(k):
                ok = valid[np.clip(k[:, 1].astype(int), 0, im.shape[0] - 1), np.clip(k[:, 0].astype(int), 0, im.shape[1] - 1)]
                k, d = k[ok], d[ok]
            F1.append((k, d)); valid_last = u
        cv2.imwrite(f'{out}/bonus_undistort_{fl}.jpg', np.hstack([imgs[len(imgs) // 2], undistort_image(imgs[len(imgs) // 2], kappa)[0]]))

        runs = {}
        for tag, F in (('original', F0), ('undistorted', F1)):
            for model in ('similarity', 'homography'):
                T, _, lost, _ = chain(F, model)
                e = gt.path_errors(path, T)
                mrows.append([fl, tag, model, len(lost), float(e.mean()), float(e[-1])])
                runs[f'{model}, {tag}'] = T
                if tag == 'undistorted':
                    np.save(f'{out}/T_{fl}_{model[:3]}_undistorted.npy', np.array(T))
        viz.plot_paths(path, runs, f'{out}/bonus_paths_{fl}.png')

        # match errors against the distance from the center, before and after
        pairs = []
        for tag, F in (('original', F0), ('undistorted', F1)):
            for i, j in ((20, 21), (3, len(F) - 6) if fl == 'loop' else (20, 22)):
                (k1, d1), (k2, d2) = F[i], F[j]
                mm = match(d1, d2)
                pairs.append((f'{fl} {i}-{j}, {tag}', k1[mm[:, 0]], k2[mm[:, 1]], gt.gt_H(path, i, j)))
        viz.plot_error_vs_radius(pairs, f'{out}/bonus_error_vs_radius_{fl}.png', max_error=15.0)
    print('\nEstimated kappa')
    utils.table(f'{out}/bonus_kappa.md', ['flight', 'pairs used', 'kappa'], krows)
    print('\nQ3.2 again, before and after undistortion')
    utils.table(f'{out}/bonus_models.md', ['flight', 'frames', 'model', 'lost steps', 'mean error (m)', 'final error (m)'], mrows)
    print(f'\nEverything is in {os.path.relpath(out)}')


if __name__ == '__main__':
    utils.run(main)
