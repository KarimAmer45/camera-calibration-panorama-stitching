import argparse
import cv2
import numpy as np
import math

np.set_printoptions(suppress=True, precision=5)


def click_event(event, x, y, flags, param):
    """OpenCV mouse callback that records left-click coordinates."""
    if not hasattr(click_event, "pts"):
        click_event.pts = []

    if event == cv2.EVENT_LBUTTONDOWN:
        click_event.pts.append((int(x), int(y)))

def pick_points(image, window_name='image'):
    """
    Generic point picker.
    """
    img = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    # Reset stored points
    click_event.pts = []

    # Minimal interactive loop: press 'q' to finish.
    while True:
        disp = img.copy()
        for (px, py) in click_event.pts:
            cv2.drawMarker(disp, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    pts_xy = np.array(click_event.pts, dtype=np.float64)
    return pts_xy

def compute_Perspective(pts_src, pts_target):
    """Compute homography H such that (x',y',1)^T ~ H (x,y,1)^T via DLT."""
    assert pts_src.shape == pts_target.shape
    n = pts_src.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 point correspondences")

    A = []
    for i in range(n):
        x, y = pts_src[i]
        u, v = pts_target[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)

    # Solve Ah=0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def compute_error(H, pts_src, pts_dst):
    pts_src_h = np.hstack([pts_src, np.ones((pts_src.shape[0], 1), dtype=np.float64)])
    proj = (H @ pts_src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    errs = np.linalg.norm(proj - pts_dst, axis=1)
    print("Alignment errors (pixels):")
    for i, e in enumerate(errs):
        print(f"  pt {i:02d}: {e:.4f}")
    print(f"Mean error: {errs.mean():.4f} px")
    return errs


def _auto_correspondences(img1, img2, max_pts=12):
    """Fallback helper: uses SIFT + cv2 RANSAC just to GET correspondences.

    The actual homography used in this task is still computed with our own DLT
    (compute_Perspective) without RANSAC.
    """
    sift = cv2.SIFT_create()
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 4:
        raise RuntimeError("Not enough matches for auto correspondences")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        raise RuntimeError("cv2.findHomography failed")
    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    # pick up to max_pts evenly
    k = min(max_pts, pts1_in.shape[0])
    if k < 4:
        raise RuntimeError("Not enough inliers")
    idx = np.linspace(0, pts1_in.shape[0] - 1, k).astype(int)
    return pts1_in[idx].astype(np.float64), pts2_in[idx].astype(np.float64)


def task_02(args):
    # Load images
    imgA = cv2.imread(args.a)
    imgB = cv2.imread(args.b)
    imgC = cv2.imread(args.c)

    # Step 1: pick point correspondences for image pairs AB and BC
    # -----------------------------------------------------------------
    # For submission: replace the following automatic correspondences with
    # hard-coded arrays from your own manual clicking (pick_points), e.g.
    # ptsA = np.array([[...],[...],...], dtype=np.float64)
    # ptsB = np.array([[...],[...],...], dtype=np.float64)
    # ptsB2 = ...
    # ptsC = ...
    ptsA, ptsB = _auto_correspondences(imgA, imgB, max_pts=10)
    ptsB2, ptsC = _auto_correspondences(imgB, imgC, max_pts=10)

    # Step 2: Compute perspective transformation between A and B
    # -----------------------------------------------------------------
    H_AB = compute_Perspective(ptsA, ptsB)
    compute_error(H_AB, ptsA, ptsB)
    warpedA_to_B = cv2.warpPerspective(imgA, H_AB, (imgB.shape[1], imgB.shape[0]))
    vis_AB = np.hstack([imgB, warpedA_to_B])
    cv2.imshow("B | warp(A->B)", vis_AB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 3: Compute perspective transformation between B and C
    # -----------------------------------------------------------------
    H_BC = compute_Perspective(ptsB2, ptsC)
    compute_error(H_BC, ptsB2, ptsC)
    warpedB_to_C = cv2.warpPerspective(imgB, H_BC, (imgC.shape[1], imgC.shape[0]))
    vis_BC = np.hstack([imgC, warpedB_to_C])
    cv2.imshow("C | warp(B->C)", vis_BC)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 4: Derive A->C transformation and visualize stitched panorama
    # -----------------------------------------------------------------
    H_AC = H_BC @ H_AB

    # Compute canvas bounds by warping corners into C coordinate frame
    def corners(w, h):
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)

    cA = corners(imgA.shape[1], imgA.shape[0])
    cB = corners(imgB.shape[1], imgB.shape[0])
    cC = corners(imgC.shape[1], imgC.shape[0])

    def warp_pts(pts, H):
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        proj = (H @ pts_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    cA_w = warp_pts(cA, H_AC)
    cB_w = warp_pts(cB, H_BC)
    cC_w = cC.copy()
    all_pts = np.vstack([cA_w, cB_w, cC_w])
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)
    shift_x = -min_xy[0] if min_xy[0] < 0 else 0
    shift_y = -min_xy[1] if min_xy[1] < 0 else 0
    out_w = int(max_xy[0] + shift_x)
    out_h = int(max_xy[1] + shift_y)

    T = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    WA = cv2.warpPerspective(imgA, T @ H_AC, (out_w, out_h))
    WB = cv2.warpPerspective(imgB, T @ H_BC, (out_w, out_h))
    WC = cv2.warpPerspective(imgC, T @ np.eye(3), (out_w, out_h))

    # simple alpha blending: average where overlapping
    acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    wgt = np.zeros((out_h, out_w, 1), dtype=np.float64)
    for W in [WA, WB, WC]:
        mask = (np.sum(W, axis=2, keepdims=True) > 0).astype(np.float64)
        acc += W.astype(np.float64) * mask
        wgt += mask
    pano = (acc / np.maximum(wgt, 1.0)).astype(np.uint8)

    cv2.imshow("Panorama (A,B,C in C frame)", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="./data/A.png", help="Image A path")
    parser.add_argument("--b", type=str, default="./data/B.png", help="Image B path")
    parser.add_argument("--c", type=str, default="./data/C.png", help="Image C path")
    args = parser.parse_args()
    task_02(args)
