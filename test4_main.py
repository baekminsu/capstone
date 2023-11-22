import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 비디오 파일 또는 웹캠에서 캡처
path = "./testroad.mp4"
cap = cv2.VideoCapture(path)  # path에 0 넣으면 연결한 웹캠 사용 가능

# 출력용 비디오 작성기
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('../output_2.mp4', codec, 30.0, (int(width), int(height)))

# 이미지 처리에 사용되는 매개변수
kernel_size = 5
low_threshold = 15
high_threshold = 100

# 허프 변환에 사용되는 매개변수
rho = 2
theta = np.pi / 180
threshold = 90
min_line_len = 100
max_line_gap = 50

# 이미지에 선을 그릴 때 사용되는 매개변수
a = 0.8
b = 1.
theta_w = 0.

# 이미지를 그레이스케일로 변환하는 함수
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 가우시안 블러를 적용하는 함수 # 노이즈제거
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Canny 엣지 검출을 적용하는 함수
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# 자동 Canny 엣지 검출을 적용하는 함수
def auto_canny(img, sigma):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)

# 관심 영역을 정의하는 함수
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 이미지에 선을 그리는 함수
def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Hough 변환 및 이미지에 선을 그리는 함수
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# 원본 이미지에 선을 오버레이하는 함수
def weighted_img(img, initial_img, a, b, theta_w):
    return cv2.addWeighted(initial_img, a, img, b, theta_w)

# 메인 루프
while cap.isOpened():
    ret, img = cap.read()
    # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        print("프레임을 수신할 수 없습니다. 종료 중 ...")
        break

    gray = grayscale(img)
    blur_gray = gaussian_blur(gray, kernel_size)

    edges = canny(blur_gray, low_threshold, high_threshold)
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    imshape = img.shape

    vertices = np.array([[(0, 1080),  # 왼쪽 아래
                         (0, 300),   # 왼쪽 위
                         (1920, 300),  # 오른쪽 위
                         (1920, 1080)]], dtype=np.int32)  # 오른쪽 아래
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = region_of_interest(edges, vertices)

    lines = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
    lines_edges = weighted_img(lines, img, a, b, theta)

    cv2.imshow('lane-detection', lines_edges)
    out.write(lines_edges)
    
    if cv2.waitKey(25) == ord('q'):
        break

# 작업 완료 후 해제
cap.release()
out.release()
cv2.destroyAllWindows()
