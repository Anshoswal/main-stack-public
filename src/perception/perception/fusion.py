#!/usr/bin/env python3

# cv2
from cv_bridge import CvBridge

# Import perception packages:
from perception.utils import *
from perception.utils.perc_utils import *
from perception.object_detect.yolo_new import Yolo
from perception.Keypoint_Detection.Keypoint import Keypoints
from perception.perception_packages.sift import Features
from perception.mono import *
from perception.utils.perc_utils import draw_images, check_cone_overlap
# from perception.process_image import ProcessImage

# Import math dependencies
from math import sin,cos,tan,atan2,sqrt,pi,exp,hypot
from cmath import rect

# Import other dependencies
import numpy as np
import yaml

class FusionDepth():
    def __init__(self, CONFIG_PATH:str, platform:str):
        
        self.platform = platform
        perc_config_path = CONFIG_PATH / 'perception_config.yaml'
        with open(perc_config_path) as file:
            perc_config = yaml.load(file, Loader=yaml.FullLoader)
        # Constants
        self.image_size = perc_config['constants_mono']['image_size']
        self.big_orange_a = perc_config['constants_mono']['profiles']['big_orange'][platform]['a']
        self.big_orange_b = perc_config['constants_mono']['profiles']['big_orange'][platform]['b']
        self.small_a = perc_config['constants_mono']['profiles']['small']['trt_v11']['a']
        self.small_b = perc_config['constants_mono']['profiles']['small']['trt_v11']['b']
        self.cx = perc_config['constants_fusion']['cx']
        self.cy = perc_config['constants_fusion']['cy']
        self.fx = perc_config['constants_fusion']['fx']
        self.fy = perc_config['constants_fusion']['fy']
        self.image_width = perc_config['constants_mono']['image_size']['width']
        self.image_height = perc_config['constants_mono']['image_size']['height']
        self.distance_threshold = perc_config['constants_fusion']['r']
        self.orientation_angle = perc_config['constants_fusion']['orientation_angle']
        self.Y_offset = perc_config['constants_fusion']['Y_offset']
    
    def get_pcd(self,msg):
        global lidar_height
        points = pc2.read_points(msg,skip_nans=True)
        points = np.array([[data[0]+0.59,data[1],data[2]] for data in points if (data[0] > 0)]) # and -lidar_height+0.35 > data[2]> -lidar_height)])
        # X = points[:, :2]  # Use x, y for plane fitting
        # y = points[:, 2]   # Use z as the target
        # ransac = RANSACRegressor()
        # ransac.fit(X, y)

        # # Predict the ground model (plane) for all points
        # ground_z_pred = ransac.predict(X)

        # # Calculate the residuals (difference between actual z and predicted z)
        # residuals = np.abs(y - ground_z_pred)

        # # Define a threshold for the residuals (points close to the ground plane)
        # threshold = 0.0
        points_filtered = points
        #self.points_filtered= np.array([[data[0],data[1],data[2]] for data in points if (data[2]>-0.2)])
        model = DBSCAN(eps=0.2, min_samples=2)
        labels = model.fit_predict(points_filtered)
        self.lidar_coords = cones_xy(points_filtered,labels)
    
    def depth_for_one_bb(self, image, bb, camera):  # finding depth for one bb 
        # bb is [class, [x,y,w,h], conf]
        cls = bb.cls.cpu().numpy()[0]
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
                depth = self.small_a * ((h / ly) ** self.small_b)

        # The following part is for fusion
        cone_middle = [xywh[0][0], xywh[0][1]]
        theta,range_3d = bearing(depth, cone_middle, image.shape)
        X = depth
        Y = -(cone_middle[0] - self.cx) * (depth / self.fx)
        Z = (cone_middle[1] - self.cy) * (depth / self.fy)

        if(camera == "Left"):

            X_est = X*np.cos(-self.orientation_angle) + Y*np.sin(-self.orientation_angle)
            Y_est = -X*np.sin(-self.orientation_angle) + Y*np.cos(-self.orientation_angle)
            Y_est = Y_est - self.Y_offset

        elif(camera == "Right"):
            X_est = X*np.cos(self.orientation_angle) - Y*np.sin(self.orientation_angle)
            Y_est = X*np.sin(self.orientation_angle) + Y*np.cos(self.orientation_angle)
            Y_est = Y_est + self.Y_offset
        
        cone_xy = np.array([X_est, Y_est, cls, 0])

        print("depth for 1 bbb retrinong`")
        return depth, theta, range_3d, cls, cone_xy
    
    def find_depth(self ,boxes, image, lidar_coords, camera):

        print("FusionDepth: find_depth")
        if len(boxes) != 0:

            depths = np.array([])
            thetas = np.array([])  # Initialize thetas
            ranges = np.array([])
            colors = np.array([])
            camera_coords = np.empty((0,4))

            for bb in boxes:
                
                depth_using_h, theta, range, cls, cone_xy = self.depth_for_one_bb(image, bb, camera)

                depths = np.append(depths, depth_using_h)
                thetas = np.append(thetas, theta)
                ranges = np.append(ranges, range)
                colors = np.append(colors, cls)
                camera_coords = np.vstack((camera_coords, cone_xy))

            camera_coords = np.sort(camera_coords, axis=0)
            fusion_depth = search2(camera_coords, lidar_coords, self.distance_threshold)
            
            depths_using_fusion = fusion_depth[:,0]
            # thetas = fusion_depth[:,1]
            # colors = fusion_depth[:,2]
            ranges = fusion_depth[:,3]
            
            pipeline_running = False
            return depths, depths_using_fusion, thetas, ranges, colors, pipeline_running


        else:
            # if len of bb array is 0
            print("NO CONES WERE DETECTED")
    
class FusionPipeline():

    def __init__(self, config_path, platform, logger):
        CONFIG_PATH = config_path
        self.platform = platform
        self.logger = logger
        self.fusion = FusionDepth(CONFIG_PATH, self.platform)
        # self.yolo = yolo
    
    def fusionpipeline(self, left_image, right_image, image_number, left_boxes, right_boxes, lidar_coords):  
        if left_image is None:
            self.logger.error("Image is None, Image number: {image_number}")
            return 
        
        # Run the mono pipeline
        # colors = [box[0].cpu().numpy() for box in boxes]
        depths_left, depths_using_fusion_left, thetas_left, ranges_left, colors_left, running_status = self.fusion.find_depth(left_boxes, left_image, lidar_coords, "Left")
        depths_right, depths_using_fusion_right, thetas_right, ranges_right, colors_right, running_status = self.fusion.find_depth(right_boxes, right_image, lidar_coords, "Right")
        
        depths = depths_left + depths_right
        depths_using_fusion = depths_using_fusion_left + depths_using_fusion_right
        thetas = thetas_left + thetas_right
        ranges = ranges_left + ranges_right
        colors = colors_left + colors_right
        draw_images(right_boxes, right_image, depths, depths_using_fusion, image_number)
        return depths, depths_using_fusion, thetas, ranges, colors