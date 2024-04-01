import cv2
import torch
import numpy as np
from super_gradients.training import models
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from utils.image_preprocess import crop_person_objects, preprocess_for_vit, save_cropped_images
# from args_parser import args
import math
import pandas as pd

classNames = [
    
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"

]

def compute_color_for_labels(label):
    """
    Compute a color for a label.
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    """
    Draw bounding boxes and labels on an image.
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        cv2.rectangle(img, (x1, y1), (x2, y2), color= compute_color_for_labels(cat),thickness=2, lineType=cv2.LINE_AA)
        label = str(id) + ":" + classNames[cat]
        (w,h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)
        t_size=cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)[0]
        c2=x1+t_size[0], y1-t_size[1]-3
        cv2.rectangle(img, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img

def save_to_csv(results, out_file_path):
    """
    Save the results to a CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(out_file_path, index=False)

def process_video(video_path, output_path, results_csv_path, tensor_results_csv_path):
    """
    Process a video file to detect and track objects, saving the output to a new video.
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (frame_width, frame_height))

    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT, max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE, nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE, max_age=100, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=150, use_cuda=True)
    results = [] # frmae 당 detection 결과 저장
    tensor_results = [] # detection bbox image -> tensor로 변환하여 저장
    count  = 0
    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break
        # Frame processing code
        xywh_bboxs = []
        confs = []
        oids = [] 
        # 모델 예측 실행
        result = list(model.predict(frame, conf=0.5, fuse_model=False))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()

        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            if cls == 0:  # "person" 클래스의 인덱스가 0
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((confidence*100))/100
                # 바운딩 박스의 중심과 크기를 계산
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                # 리스트에 추가
                xywh_bboxs.append([cx, cy, bbox_width, bbox_height])

                confs.append(conf)
                oids.append(cls)

        # Tensor로 변환
        if xywh_bboxs:
            xywhs = torch.tensor(xywh_bboxs, dtype=torch.float32)
            confss = torch.tensor(confs, dtype=torch.float32)
            # DeepSort 업데이트
            outputs = deepsort.update(xywhs, confss, oids, frame)
            person_count = len(outputs)
            if person_count > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                cropped_images = crop_person_objects(frame, bbox_xyxy)
                # 매 프레임마다 cropped_images를 저장
                save_cropped_images(cropped_images, count, f"yolo_save_img")
                preprocessed_images = preprocess_for_vit(cropped_images,target_size=224)
                # preprocessed_images들을 별도의 csv 파일로 저장
                tensor_results.append({'frame_number': count, 'person_count': person_count, 'preprocessed_images': preprocessed_images})
                frame = draw_boxes(frame, bbox_xyxy, identities, object_id, classNames)                
                # 각 객체의 bbox 좌표와 object_id를 출력
                for i, (bbox, id) in enumerate(zip(bbox_xyxy, identities)):
                    # draw_boxes 함수에서의 bbox 좌표
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]            
                    print(f"Frame N: {count}, Person N: {person_count} Object ID: {id}, BBOX: ({x1}, {y1}, {x2}, {y2})")
                    results.append({'frame_number': count, 'person_count': person_count, 'object_id': id, 'bbox': (x1, y1, x2, y2)})
            else:
                print(f"Frame N: {count} Person N: 0")
        else:
            print(f"Frame N: {count} Person N: 0")
        
        output.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_to_csv(results, results_csv_path)
    save_to_csv(tensor_results, tensor_results_csv_path)
    output.release()
    cap.release()
    cv2.destroyAllWindows()