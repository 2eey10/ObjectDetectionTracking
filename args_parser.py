import argparse
from utils.tracking import process_video
# 인스턴스 생성

def main():
    parser = argparse.ArgumentParser()

    # 입력받을 인자 추가
    parser.add_argument('--video_path', type=str, help='Path to the input video file', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save the output video file', required=True)
    parser.add_argument('--results_csv_path', type=str, help='Path to save the results CSV file', required=True)
    parser.add_argument('--tensor_results_csv_path', type=str, help='Path to save the tensor results CSV file', required=True)
    
    # 입력받은 인자값을 args에 저장
    args = parser.parse_args()
    deepsort_params = {
        'reid_ckpt': 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
        'max_dist': 0.2,
        'min_confidence': 0.3,
        'nms_max_overlap': 1.0,
        'max_iou_distance': 0.7,
        'max_age': 70,
        'n_init': 3,
        'nn_budget': 100,
        'use_cuda': True
    }
    #process_video(deepsort_params=deepsort_params)

# if __name__ == "__main__":
    # main()