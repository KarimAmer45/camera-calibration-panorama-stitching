import numpy as np
import cv2
import random

np.set_printoptions(suppress=True)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2):
    """Compute a homography with a custom RANSAC (no cv2.findHomography).

    Args:
        good_matches: list[cv2.DMatch]
        kp_1, kp_2: keypoints of image_1 and image_2
    Returns:
        best_H: 3x3 homography mapping image_1 -> image_2
    """
    if len(good_matches) < 4:
        raise ValueError("Need at least 4 matches")

    def dlt_homography(p1, p2):
        # p1, p2: (N,2)
        n = p1.shape[0]
        A = []
        for i in range(n):
            x, y = p1[i]
            u, v = p2[i]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A = np.asarray(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    def project(pts, H):
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        q = (H @ pts_h.T).T
        q = q[:, :2] / q[:, 2:3]
        return q

    pts1 = np.float64([kp_1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float64([kp_2[m.trainIdx].pt for m in good_matches])

    # RANSAC parameters
    max_iters = 2000
    inlier_thr = 5.0
    best_inliers = None
    best_H = None
    best_count = -1

    idx_all = list(range(len(good_matches)))
    for _ in range(max_iters):
        sample = random.sample(idx_all, 4)
        H = dlt_homography(pts1[sample], pts2[sample])
        pred = project(pts1, H)
        err = np.linalg.norm(pred - pts2, axis=1)
        inliers = err < inlier_thr
        cnt = int(np.sum(inliers))
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_H = H
            # early exit if very good
            if best_count > 0.85 * len(good_matches):
                break

    if best_inliers is None or best_count < 4:
        # fallback to DLT over all matches
        return dlt_homography(pts1, pts2)

    # refine with all inliers
    best_H = dlt_homography(pts1[best_inliers], pts2[best_inliers])
    return best_H


def get_best_matches(des_1, des_2, thr=0.3):
    """Symmetric best-match ratio test with mutual consistency.

    Uses squared Euclidean distances and keeps only matches that pass
    the ratio test in both directions.
    """
    if des_1 is None or des_2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # forward 1 -> 2
    knn12 = bf.knnMatch(des_1, des_2, k=2)
    good12 = {}
    for m, n in knn12:
        if (m.distance * m.distance) < (thr * thr) * (n.distance * n.distance):
            good12[m.queryIdx] = m  # best for this query

    # backward 2 -> 1
    knn21 = bf.knnMatch(des_2, des_1, k=2)
    good21 = {}
    for m, n in knn21:
        if (m.distance * m.distance) < (thr * thr) * (n.distance * n.distance):
            good21[m.queryIdx] = m

    # mutual consistency
    good_matches = []
    for qidx, m in good12.items():
        tidx = m.trainIdx
        if tidx in good21 and good21[tidx].trainIdx == qidx:
            good_matches.append(m)

    good_matches.sort(key=lambda mm: mm.distance)
    return good_matches


def stitch_multiple_images(images, homographies):
    """Stitch multiple images into a panorama.

    Args:
        images: list of BGR images
        homographies: list of 3x3 homographies mapping each image -> reference
    """
    assert len(images) == len(homographies)

    def corners(w, h):
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)

    def warp_pts(pts, H):
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        q = (H @ pts_h.T).T
        q = q[:, :2] / q[:, 2:3]
        return q

    all_warped = []
    for img, H in zip(images, homographies):
        c = corners(img.shape[1], img.shape[0])
        all_warped.append(warp_pts(c, H))
    all_pts = np.vstack(all_warped)
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)

    shift_x = -min_xy[0] if min_xy[0] < 0 else 0
    shift_y = -min_xy[1] if min_xy[1] < 0 else 0
    out_w = int(max_xy[0] + shift_x)
    out_h = int(max_xy[1] + shift_y)

    T = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]], dtype=np.float64)

    acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    wgt = np.zeros((out_h, out_w, 1), dtype=np.float64)
    for img, H in zip(images, homographies):
        W = cv2.warpPerspective(img, T @ H, (out_w, out_h))
        mask = (np.sum(W, axis=2, keepdims=True) > 0).astype(np.float64)
        acc += W.astype(np.float64) * mask
        wgt += mask

    result = (acc / np.maximum(wgt, 1.0)).astype(np.uint8)
    return result


def task_03():
    # Load all images
    image_paths = [r'./data/Fuji_1.png', r'./data/Fuji_2.png', r'./data/Fuji_3.png']
    images = [cv2.imread(p) for p in image_paths]
    num_imgs = len(images)

    sift = cv2.SIFT_create()
    kps = []
    dess = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        kps.append(kp)
        dess.append(des)

    # Compute matches between all pairs and visualize them
    matches_stats = np.zeros((num_imgs, num_imgs), dtype=int)
    pair_matches = {}
    for i in range(num_imgs):
        for j in range(i + 1, num_imgs):
            print(f"Processing pair {i} - {j}")
            good_matches = get_best_matches(dess[i], dess[j], thr=0.3)
            pair_matches[(i, j)] = good_matches
            matches_stats[i, j] = len(good_matches)
            matches_stats[j, i] = len(good_matches)

            # Visualize matches
            knn = [[m] for m in good_matches]
            vis = cv2.drawMatchesKnn(images[i], kps[i], images[j], kps[j], knn, None, flags=2)
            cv2.imwrite(f"matches_{i}_{j}.png", vis)

    # Find optimal reference image: maximize total matches to others
    ref_idx = int(np.argmax(np.sum(matches_stats, axis=1)))
    print(f"Selected reference image index: {ref_idx}")

    # Compute homographies to reference via RANSAC
    homographies = [np.eye(3, dtype=np.float64) for _ in range(num_imgs)]
    for i in range(num_imgs):
        if i == ref_idx:
            continue

        # decide direction based on stored pairs
        if i < ref_idx:
            good = pair_matches.get((i, ref_idx), [])
            H = compute_Homography_RANSAC(good, kps[i], kps[ref_idx], images[i], images[ref_idx])
            homographies[i] = H
        else:
            good = pair_matches.get((ref_idx, i), [])
            H_ref_to_i = compute_Homography_RANSAC(good, kps[ref_idx], kps[i], images[ref_idx], images[i])
            # invert to map i -> ref
            homographies[i] = np.linalg.inv(H_ref_to_i)

    print("Creating final panorama...")
    final_panorama = stitch_multiple_images(images, homographies)
    cv2.imwrite("panorama.png", final_panorama)


if __name__ == "__main__":
    task_03()