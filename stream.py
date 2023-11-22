from flask import Flask, render_template, Response
from time import sleep
import cv2

app = Flask(__name__)
capture = cv2.VideoCapture(0)  # 웹캠으로부터 비디오 캡처 객체 생성
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처된 비디오의 폭 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처된 비디오의 높이 설정

def GenerateFrames():
    while True:
        sleep(0.1)  # 프레임 생성 간격을 잠시 지연시킵니다.
        ref, frame = capture.read()  # 비디오 프레임을 읽어옵니다.
        if not ref:  # 비디오 프레임을 제대로 읽어오지 못했다면 반복문을 종료합니다.
            break
        else:
            ref, buffer = cv2.imencode('.jpg', frame)  # JPEG 형식으로 이미지를 인코딩합니다.
            frame = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
            # multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def Index():
    return render_template('index.html')  # index.html 파일을 렌더링하여 반환합니다.

@app.route('/stream')
def Stream():
    # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)