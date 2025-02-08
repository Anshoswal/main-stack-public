
# Import perception packages:
# from perception.object_detect.yolo_new import Yolo
from ultralytics import YOLO

# Import other dependencies
import numpy as np
from pathlib import Path
import yaml
import cv2


class LoadYolo():

    def __init__(self, PACKAGE_ROOT:str):
        
        ROOT = Path(PACKAGE_ROOT)
        perc_config = ROOT / 'config/perception_config.yaml'
        with open(perc_config, 'r') as file:    # path to yolo yaml file
            perc_config = yaml.safe_load(file)

        self.yolo_weights = perc_config['yolo_paths']['yolo_weights_path']
        self.yolo_data = perc_config['yolo_paths']['yolo_data_path']
        self.kpr_path = perc_config['yolo_paths']['kpr_path']
        self.conf_thresh = perc_config['yolo_paths']['conf_thresh'] 
        # self.yolo_model = Yolo(weights=self.yolo_weights, data=self.yolo_data, imgsz=perc_config['yolo_paths']['image_size'], device='0')
        self.yolo_model = YOLO(self.yolo_weights)

    def make_bounding_boxes(self, image):
         
         # Detect bounding boxes using Yolo
        # boxes = self.yolo_model.detect_all(image, conf_thres = self.conf_thresh)[0]

        image = image[:, :, :3]
        detected_boxes = self.yolo_model.predict(image)[0]
        boxes = [box.boxes for box in detected_boxes]

        return boxes
