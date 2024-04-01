import cv2
import torch
from super_gradients.training import models
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)

cap = cv2.VideoCapture('/Users/leeyongryull/Desktop/dev/PIA/Task/test_video.mp4')
assert cap.isOpened(), "Error reading video file"

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output_NAS.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (frame_width, frame_height))

# COCO 데이터셋에서 'person' 클래스의 인덱스; 일반적으로 0입니다.
person_class_index = 0
count = 0

while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        result = list(model.predict(frame, conf=0.5, fuse_model=False))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            if cls == person_class_index:  # 'person' 클래스에 해당하는 객체만 처리
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = map(int, bbox)
                conf = math.ceil((confidence*100))/100
                label = f'person {conf}'
                print("Frame N", count, "", x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color=(0, 0, 255), thickness=-1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], thickness=1)
                
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
