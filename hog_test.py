import numpy as np
import cv2
import pandas as pd  

# HOG 기술자 및 기본 사람 탐지기 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 출력 비디오 설정
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

# 비디오 파일 경로 지정
video_path = '/your_video_path.mp4'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# bbox 값 저장을 위한 빈 리스트 초기화
bbox_values = []

while True:
    ret, frame = cap.read()

    # 비디오 끝에 도달하면 종료
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    # 사람 객체 탐지
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        bbox_values.append([xA, yA, xB, yB])  # bbox 값을 리스트에 추가

    out.write(frame.astype('uint8'))
    cv2.imshow('frame', frame)

    # 실시간 bbox 값을 출력 (필요에 따라 주석 처리 가능)
    print(boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CSV 파일로 저장
bbox_df = pd.DataFrame(bbox_values, columns=['x_start', 'y_start', 'x_end', 'y_end'])
bbox_df.to_csv('your_csv_path', index=False)



# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
