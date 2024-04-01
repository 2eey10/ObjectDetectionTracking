from utils.tracking import process_video

def main():
    video_path = 'input/test_video.mp4'
    output_path = 'output/output_1.avi'
    results_csv_path = 'yolo_save_csv/save_csv_1'
    tensor_results_csv_path = 'yolo_save_csv/save_csv_tensor_1'
    process_video(video_path, output_path, results_csv_path, tensor_results_csv_path) 
    
if __name__ == "__main__":
    main()    