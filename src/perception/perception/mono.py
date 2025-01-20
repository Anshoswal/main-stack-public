#!/usr/bin/env python3

# cv2
from cv_bridge import CvBridge

# Import perception packages:
from perception.perception_packages.view_utils import *
from perception.utils import *
from perception.object_detect.yolo_new import Yolo
from perception.Keypoint_Detection.Keypoint import Keypoints
from perception.perception_packages.sift import Features
# from perception.process_image import ProcessImage

# Import math dependencies
from math import sin,cos,tan,atan2,sqrt,pi,exp,hypot
from cmath import rect

# Import other dependencies
import numpy as np
import yaml


class MonoDepth():
    def __init__(self, CONFIG_PATH:str, platform:str):

        self.platform = platform
        perc_config_path = CONFIG_PATH / 'perception_config.yaml'
        with open(perc_config_path) as file:
            perc_config = yaml.load(file, Loader=yaml.FullLoader)
        # Constants
        self.image_size = perc_config['constants_mono']['image_size']
        self.big_orange_a = perc_config['constants_mono']['profiles']['big_orange'][platform]['a']
        self.big_orange_b = perc_config['constants_mono']['profiles']['big_orange'][platform]['b']
        self.small_a = perc_config['constants_mono']['profiles']['small'][platform]['a']
        self.small_b = perc_config['constants_mono']['profiles']['small'][platform]['b']

    def depth_for_one_bb(self, image, bb):  # finding depth for one bb 
        # bb is [class, [x,y,w,h], conf]
        cls = bb.cls.cpu().numpy()
        xywh = bb.xywh.cpu().numpy()
        # conf = bb[0][2]

        h = xywh[0][3]
        w = xywh[0][2]
        bby=xywh[0][1]
        lx= image.shape[1]
        ly= image.shape[0]
                
        # a bad cone is one in which the cone is half cut or the cone is fallen and thus the width is more then height
        # for bad cones, apply the stereo pipeline instead of the mono using bb height pipeline
        bad_cone = int((bby + h / 2) * ly) > ly or (w * lx) > (h * ly)
        
        if bad_cone and False:
            # have to call stereo function for bad cones
            left_boxes_new = []

            left_boxes_new = np.array(left_boxes_new)
            left_boxes_new = np.append(left_boxes_new,left_boxes[i])
            depth_using_h = sift_bypass(left_boxes_new,left_image,kpr_model,right_image)

        else:
            if cls == 1:
                depth = self.big_orange_a * ((h / ly) ** self.big_orange_b)
            else:
                depth = self.small_a * ((1.1 * h / ly) ** self.small_b)

        cone_middle = [xywh[0][0] , xywh[0][1] ]
        theta,range_3d = bearing(depth,cone_middle, image.shape)
        return depth, theta, range_3d, cls


    def find_depth(self ,boxes, image):

        if len(boxes) != 0:

            depths = np.array([])
            thetas = np.array([])  # Initialize thetas
            ranges = np.array([])
            colors = np.array([])

            for bb in boxes:
                
                depth_using_h, theta, range, cls = self.depth_for_one_bb(image,bb)

                depths = np.append(depths, depth_using_h)
                thetas = np.append(thetas, theta)
                ranges = np.append(ranges, range)
                colors = np.append(colors, cls)

            pipeline_running = False
            return depths, thetas, ranges, colors, pipeline_running


        else:
            # if len of bb array is 0
            print("NO CONES WERE DETECTED")



class MonoPipeline():

    def __init__(self, config_path, platform, logger):
        
        CONFIG_PATH = config_path
        self.platform = platform
        self.logger = logger
        self.mono = MonoDepth(CONFIG_PATH, self.platform)
        # self.yolo = yolo


    def monopipeline(self, left_image, image_number, boxes):  

        # self.logger.info(f"Processing IMAGE {image_number}")     # For debugging or synchronization if multi threading is implemented 
        
        # Check if image conversion and processing is successful, and the image is not None:
        if left_image is None:
            self.logger.error("Image is None, Image number: {image_number}")
            return 
        
        # Run the mono pipeline
        # colors = [box[0].cpu().numpy() for box in boxes]
        depths, thetas, ranges, colors, running_status = self.mono.find_depth(boxes, left_image)

        self.logger.info(f"Number of cones detected: {len(boxes)}, Image: {image_number}")
        return depths, thetas, ranges, colors