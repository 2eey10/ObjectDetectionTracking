## Human Object Detection Tracking Input Vectorization for ViT


#### A task that involves real-time object detection of person objects, conversion into a specified input format, and then feeding into ViT (Vision Transformer) as input:
* Detect person objects with bounding box detection.
* Track the detected person objects in real-time.
* Listen in real-time, and simultaneously, have the ViT also listen.

---
| Inference | Def | 
|----------|----------|
| *hog_test.py*   | Setup of HOG Descriptor and Default person Detector| 
| *webcam_test.py*   | Setup of person Object Detection Using Built-in Webcam   | 
| *yolo_main.py*   | Setup of person Object Detection Using YOLO_NAS, Tracking TOOL Deepsort   |    

---

| Utils | Def | 
|----------|----------|
| *image_preprocessing.py*   | Crop the detected person object, save it in *_.jpg_* format, and convert it into a vector suitable for input into a Vision Transformer | 
| *tracking.py*   | Utilize YOLO for detecting person objects and employ DeepSort for real-time tracking  | 

---


## Directory Achitecture

```
.
├── README.md
├── args_parser.py
├── deep_sort_pytorch
│   ├── LICENSE
│   ├── README.md
│   ├── configs
│   │   └── deep_sort.yaml
│   ├── deep_sort
│   │   ├── README.md
│   │   ├── deep
│   │   │   ├── checkpoint
│   │   │   ├── evaluate.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── model.py
│   │   │   ├── original_model.py
│   │   │   ├── test.py
│   │   │   ├── train.jpg
│   │   │   └── train.py
│   │   ├── deep_sort.py
│   │   ├── sort
│   │   │   ├── detection.py
│   │   │   ├── iou_matching.py
│   │   │   ├── kalman_filter.py
│   │   │   ├── linear_assignment.py
│   │   │   ├── nn_matching.py
│   │   │   ├── preprocessing.py
│   │   │   ├── track.py
│   │   │   └── tracker.py
│   │   └── sort - Copy
│   │       ├── iou_matching.py
│   │       ├── kalman_filter.py
│   │       ├── linear_assignment.py
│   │       ├── nn_matching.py
│   │       └── preprocessing.py
│   └── utils
│       ├── asserts.py
│       ├── draw.py
│       ├── evaluation.py
│       ├── io.py
│       ├── json_logger.py
│       ├── log.py
│       ├── parser.py
│       └── tools.py
├── input
├── output
├── ultralytics
│   ├── docker
│   ├── docs
│   ├── examples
│   ├── mkdocs.yml
│   ├── pyproject.toml
│   ├── tests
│   └── ultralytics
├── utils
│   ├── image_preprocess.py
│   └── tracking.py
├── webcam_test.py
├── yolo_main.py
├── yolo_nas.py
├── yolo_pt
├── yolo_save_csv
└── yolo_save_img
```

