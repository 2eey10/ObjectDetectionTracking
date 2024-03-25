import cv2
import torch
import pandas as pd

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# 비디오 파일 경로
video_path = 'your_video_path.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

frame_number = 0
detection_summary = []  # 프레임 번호와 탐지된 사람의 수를 저장할 리스트

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5를 사용하여 프레임에서 객체 탐지
    results = model(frame)
    person_count = 0  # 감지된 사람 수를 저장하기 위한 변수 초기화

    for *xyxy, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] == 'person':
            person_count += 1
            x1, y1, x2, y2 = map(int, xyxy)
            # 탐지된 사람 객체에 대해 bbox 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(f"Frame {frame_number}: {person_count} person(s) detected")
    detection_summary.append([frame_number, person_count])

    frame_number += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 탐지 요약을 CSV 파일로 저장
save_path = 'your_csv_path'  # 파일 확장자를 포함하도록 수정
df = pd.DataFrame(detection_summary, columns=['Frame Number', 'Person Count'])
df.to_csv(save_path, index=False)
