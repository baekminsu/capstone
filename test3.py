import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 비디오 파일 경로
path = "./park2.mp4"
cap = cv2.VideoCapture(path)  # 0을 사용하여 연결된 웹캠을 사용할 수 있습니다.

# 비디오 속성
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('../output_2.mp4', codec, 30.0, (int(width), int(height)))

# 이미지 처리를 위한 매개변수
kernel_size = 5
low_threshold = 50
high_threshold = 100

# 허프 변환 매개변수
rho = 2
theta = np.pi / 180
threshold = 200
min_line_len = 120
max_line_gap = 150

# 차선을 결합하는 데 사용되는 가중 평균 매개변수
a = 0.8
b = 1.
theta_w = 0.

# 이미지를 그레이스케일로 변환하는 함수
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 이미지에 가우시안 블러를 적용하는 함수
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# 이미지에 캐니 에지 검출을 적용하는 함수
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# 관심 영역을 설정하는 함수
def roi(img, h, w):
    mask = np.zeros_like(img)
    vertices = np.array([[(0, h), (0, h * 1 / 2), (w, h * 1 / 2), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

# 기울기 각도를 제한하는 함수
def restrict_deg(lines, min_slope, max_slope):
    slope_deg = np.rad2deg(np.arctan2(lines[:, 1] - lines[:, 3], lines[:, 0] - lines[:, 2]))
    lines = lines[np.abs(slope_deg) < max_slope]
    slope_deg = slope_deg[np.abs(slope_deg) < max_slope]
    lines = lines[np.abs(slope_deg) > min_slope]
    slope_deg = slope_deg[np.abs(slope_deg) > min_slope]
    return lines, slope_deg

# 좌우 차선을 분리하는 함수
def separate_line(lines, slope_deg):
    l_lines, r_lines = lines[(slope_deg > 0), :], lines[(slope_deg < 0), :]
    l_slopes, r_slopes = slope_deg[(slope_deg > 0)], slope_deg[(slope_deg < 0)]
    if len(l_lines) == 0 or len(r_lines) == 0:
        return
    l_line = [sum(l_lines[:, 0]) / len(l_lines), sum(l_lines[:, 1]) / len(l_lines),
              sum(l_lines[:, 2]) / len(l_lines), sum(l_lines[:, 3]) / len(l_lines)]
    r_line = [sum(r_lines[:, 0]) / len(r_lines), sum(r_lines[:, 1]) / len(r_lines),
              sum(r_lines[:, 2]) / len(r_lines), sum(r_lines[:, 3]) / len(r_lines)]
    l_slope = int(sum(l_slopes) / len(l_slopes))
    r_slope = int(sum(r_slopes) / len(r_slopes))
    return l_line, r_line, l_slope, r_slope

# 허프 변환을 적용하는 함수
def hough(img, h, w, min_line_len, min_slope, max_slope):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=30, minLineLength=min_line_len,
                            maxLineGap=30)  # 반환 값 = [[x1, y1, x2, y2], [...], ...]
    lines = np.squeeze(lines)
    lanes, slopes = restrict_deg(lines, min_slope, max_slope)
    if separate_line(lanes, slopes) != None:
        l_lane, r_lane, l_slope, r_slope = separate_line(lanes, slopes)
    else:
        l_lane, r_lane, l_slope, r_slope = [0, 0, 0, 0]
    return l_lane, r_lane, l_slope, r_slope

# 차선 감지를 수행하는 함수
def lane_detection(min_line_len, min_slope, max_slope):
    ret, img = cap.read()
    gray_img = grayscale(img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 50, 200)
    roi_img = roi(canny_img, height, width)
    l_lane, r_lane, l_slope, r_slope = hough(roi_img, height, width, min_line_len, min_slope, max_slope)
    steer_value = l_slope + r_slope
    if l_lane != 0:
        cv2.line(img, (int(l_lane[0]), int(l_lane[1])), (int(l_lane[2]), int(l_lane[3])), color=[0, 0, 255],
                 thickness=5)
        cv2.line(img, (int(r_lane[0]), int(r_lane[1])), (int(r_lane[2]), int(r_lane[3])), color=[255, 0, 0],
                 thickness=5)
    return img, steer_value

# 트랙바 콜백 함수
def nothing(pos):
    pass

# 비디오가 열려 있으면 계속 진행
while cap.isOpened():
    result_img, steer_value = lane_detection(min_line_len, 100, 200)
    cv2.imshow('차선 감지', result_img)

    out.write(result_img)

    if cv2.waitKey(25) == ord('q'):
        break

# 완료 후 해제
cap.release()
out.release()
cv2.destroyAllWindows()
