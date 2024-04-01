from torchvision import transforms
import cv2 
import os

def crop_person_objects(frame, bbox_xyxy):
    """
    주어진 프레임에서 person objects의 bounding box 영역에 해당하는 이미지를 잘라내어 반환
    
    Parameters:
    - frame: 원본 이미지 프레임
    - bbox_xyxy: bounding box 좌표의 리스트. 각 bounding box는 [x1, y1, x2, y2] 형식입니다.
    
    Returns:
    - crops: 잘라낸 이미지들의 리스트
    """
    crops = []
    for bbox in bbox_xyxy:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # 매 프레임마다 객체의 bounding box를 crop하여 저장
        crop_img = frame[y1:y2, x1:x2]
        crops.append(crop_img)
    return crops

def save_cropped_images(cropped_images, frame_number, output_dir):
    """
    잘라낸 이미지들을 디렉토리에 저장
    
    Parameters:
    - cropped_images: 잘라낸 이미지들의 리스트
    - frame_number: 현재 프레임 번호
    - output_dir: 저장할 디렉토리 경로
    """
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, img in enumerate(cropped_images):
        # 고유 파일 이름 생성을 위해 프레임 번호와 인덱스를 포함
        file_path = f"{output_dir}/frame_{frame_number}_cropped_{i}.jpg"
        cv2.imwrite(file_path, img)

def preprocess_for_vit(cropped_images, target_size=224):
    """
    ViT의 입력형식으로 이미지 사전 처리
    
    Parameters:
    - cropped_images: 잘라낸 이미지들의 리스트
    - target_size: ViT 입력에 필요한 이미지 크기
    
    Returns:
    - preprocessed_images: 사전 처리된 이미지의 텐서 리스트
    """
    preprocessed_images = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img in cropped_images:
        # 이미지를 ViT 입력 형식으로 변환
        img_tensor = transform(img)
        preprocessed_images.append(img_tensor)
    
    return preprocessed_images 