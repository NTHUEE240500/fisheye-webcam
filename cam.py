import cv2
assert cv2.__version__[0] >= '3', 'requires opencv version >= 3'
import time
import numpy as np
import glob
import os


def help():
    print('This is fisheye camera streaming')
    print('keys:')
    print('\t`h`  help')
    print('\t`q`  quit')
    print('\t`s`  save current frame')
    print('\t`x`  save current frame for calibration')
    print('\t`a`  tuggle origin and undistorted mode')
    print('\t`c`  calibrate camera with chessboard')


def save_img(img, img_dir):
    img_name = "frame" + str(int(time.time())) + ".jpg"
    cv2.imwrite(os.path.join(img_dir, img_name), img)
    print("%s saved in %s/!" % (img_name, img_dir))


def calibrate_fisheye(img_path, param_path):
    CHECKERBOARD = (6, 9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
        cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[
        0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(img_path + '/*')
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[
                :2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))

    np.save(param_path['DIM'], _img_shape)
    np.save(param_path['K'], K)
    np.save(param_path['D'], D)


def load_fisheye_param(param_path):
    DIM = tuple(np.load(param_path['DIM']))
    K = np.load(param_path['K'])
    D = np.load(param_path['D'])
    return DIM, K, D


def undistort_img(img, paramter):
    DIM, K, D = paramter
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def streaming(dir_path, file_path):
    paramter = None
    flag = 0
    while True:
        try:
            ret, frame = cam.read()
        except:
            raise Exception('read camera failed')

        if flag == -1:
            if paramter is None:
                paramter = load_fisheye_param(file_path)
            frame = undistort_img(frame, paramter)
        cv2.imshow("test", frame)

        k = chr(cv2.waitKey(1) & 0xFF)
        if k == 'q':
            break
        elif k == 'h':
            help()
        elif k == 'c':
            calibrate_fisheye('calib_img', file_path)
            paramter = None
        elif k == 'a':
            check = [os.path.exists(i) for i in file_path.values()]
            if False in check:
                print('please calibrate fisheye first')
            else:
                flag = ~flag
        elif k == 's':
            save_img(frame, dir_path['img'])
        elif k == 'x':
            save_img(frame, dir_path['calib'])

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture images")
    print('press `h` to get help')

    dir_path = {
        'img': 'img',
        'calib': 'calib_img',
        'param': 'parameter'
    }
    for name in dir_path.values():
        if not os.path.exists(name):
            os.makedirs(name)

    file_path = {
        'DIM': os.path.join(dir_path['param'], 'dim.npy'),
        'K': os.path.join(dir_path['param'], 'k.npy'),
        'D': os.path.join(dir_path['param'], 'd.npy')
    }

    streaming(dir_path, file_path)

    cam.release()
    cv2.destroyAllWindows()
