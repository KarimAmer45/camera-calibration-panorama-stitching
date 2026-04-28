import numpy as np
import cv2
import glob

np.set_printoptions(suppress=True)

def task_01():
    # Load calibration images from the local checkerboard image folder.
    patterns = [
        "./camera_calibration/*.png",
        "./camera_calibration/*.jpg",
        "./camera_calibration/*.jpeg",
        "./camera_calibration/*.bmp",
    ]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print("No calibration images found in ./camera_calibration/.")
        print("Please add your own checkerboard photos there (png/jpg/jpeg/bmp) and re-run.")
        return

    # Try a few common inner-corner sizes and pick the one that yields most detections.
    candidate_sizes = [(9, 6), (8, 6), (7, 6), (10, 7), (11, 8)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    # Physical square size (meters). Change to your real checkerboard square size.
    square_size = 0.024  # 24mm

    best_size = None
    best_hits = -1
    best_hits_mask = None

    # Count detections per candidate
    for (nx, ny) in candidate_sizes:
        hits = 0
        hits_mask = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                hits_mask.append(False)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCorners(gray, (nx, ny))
            hits_mask.append(bool(found))
            if found:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_size = (nx, ny)
            best_hits_mask = hits_mask

    if best_size is None or best_hits < 3:
        print("Could not reliably detect a checkerboard in enough images.")
        print("Try changing candidate_sizes or ensure images clearly show the checkerboard.")
        return

    print(f"Using checkerboard inner-corner size: {best_size} (detections: {best_hits}/{len(image_paths)})")
    nx, ny = best_size

    # Prepare object points in real scale (Z=0 plane)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints = []
    imgpoints = []

    img_shape = None
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (w, h)

        found, corners = cv2.findChessboardCorners(gray, (nx, ny))
        if not found:
            continue

        corners_ref = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners_ref)

    if len(objpoints) < 3:
        print("Not enough valid calibration views after refinement.")
        return

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n=== Calibration Results ===")
    print(f"RMS reprojection error (OpenCV): {ret:.6f}")
    print("\nCamera matrix K:")
    print(K)
    print("\nDistortion coefficients [k1 k2 p1 p2 k3 ...]:")
    print(dist.ravel())

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print("\nDerived intrinsics:")
    print(f"fx = {fx:.3f}, fy = {fy:.3f}, cx = {cx:.3f}, cy = {cy:.3f}")

    total_err = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        n = len(imgpoints2)
        total_err += (err * err)
        total_points += n

    mean_err = np.sqrt(total_err / max(total_points, 1))
    print(f"\nMean reprojection error (pixels): {mean_err:.6f}")
    print("Note: object points use your square_size, so extrinsics are in real units; reprojection error is in pixels.")

if __name__ == "__main__":
    task_01()
