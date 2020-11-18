import cv2
import numpy as np
import datetime

def _get_configs(path_to_file='config.yml'):
    from yaml import load, Loader
    with open(path_to_file, 'rt') as f:
        data = load(f, Loader=Loader)
    return data['feature_params'], data['lk_params'], data['diff_limit'], data['brightness_limit']


def _start_video():
    cap_ = cv2.VideoCapture(0)
    if not cap_:
        raise Exception("Cannot start camera")
    else:
        return cap_


def _get_frame(cap):
    ret, frame = cap.read()
    return ret, frame


def calculate_optical_flow(old_gray, frame_gray, old_points, diff_limit, **kwargs):
    movement_detected = False
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **kwargs)
    good_new = p1[st == 1]
    good_old = old_points[st == 1]
    if good_old.size == 0 or good_new.size == 0:
        return False, good_old, good_new
    new_mag, _ = cv2.cartToPolar(good_new[..., 0], good_new[..., 1])
    old_mag, _ = cv2.cartToPolar(good_old[..., 0], good_old[..., 1])
    if np.any(np.abs(new_mag - old_mag) > diff_limit):
        movement_detected = True
    return movement_detected, good_old, good_new


def construct_trajectory(good_new, good_old, mask, color, first_call):

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        if first_call:
            mask = cv2.circle(mask, (a, b), 5, color[i].tolist(), -1)
    return mask


def reset_points(frame):
    return np.zeros_like(frame)


if __name__ == '__main__':
    cap = _start_video()
    feature_params, lk_params, diff_limit, brightness_limit = _get_configs()
    color = np.random.randint(0, 255, (feature_params['maxCorners'], 3))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, old_frame = _get_frame(cap)
    if ret:
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_yuv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2YUV)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        size = (int(cap.get(3)), int(cap.get(4)))
        result = cv2.VideoWriter('output_without_blanks.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        result_blanks = cv2.VideoWriter('output_with_blanks.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    else:
        exit(0x01)

    while True:
        ret, frame = _get_frame(cap)
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            movement_detected, good_old, good_new = calculate_optical_flow(old_gray, frame_gray,
                                                                           p0, diff_limit, **lk_params)
            if movement_detected and \
                    np.abs(np.average(frame_yuv[..., 0]) - np.average(old_yuv[..., 0])) < brightness_limit:
                cv2.putText(frame, str(datetime.datetime.now()), (10, 25), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Feedback', frame)
                result.write(frame)
                result_blanks.write(frame)
            else:
                frame = cv2.line(np.zeros_like(frame), (0, 0), (frame.shape[1], frame.shape[0]), [0, 0, 255], 10)
                frame = cv2.line(frame, (0, frame.shape[0]), (frame.shape[1], 0), [0, 0, 255], 10)
                cv2.putText(frame, str(datetime.datetime.now()), (10, 25), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Feedback', frame)
                result_blanks.write(frame)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
            old_gray = frame_gray.copy()
            old_yuv = frame_yuv.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    cap.release()
    result.release()
    cv2.destroyAllWindows()
