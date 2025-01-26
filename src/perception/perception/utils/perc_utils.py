import cv2
from cv_bridge import CvBridge
import numpy as np

# Import perception packages:
from perception.perception_packages.view_utils import *


# def draw_images(left_boxes , left_image , data_to_write, package_path):
        
#         for i, (cls,xywh,conf) in enumerate(left_boxes):
#             lx= left_image.shape[1]
#             ly=left_image.shape[0]
#             x,y,w,h = xywh
#             x1 = int((x - w / 2) * lx)      
#             y1 = int((y - h/2 ) * ly)
#             x2 = int((x + w / 2) * lx)
#             y2 = int((y + h / 2) * ly)
#             cv2.rectangle(left_image, (x1,y1), (x2,y2), (70,255,255), 1)

#             data = f"{np.round(data_to_write[i],2)} "
#             cv2.putText(left_image, data , (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100,255,100), 1, cv2.LINE_AA) 
        
#         path = package_path / 'run_images_ros/camleft{0}.png'
#         cv2.imwrite(path , left_image)  ## this path shld come from config file

def draw_images(left_boxes , left_image , data_to_write, package_path = None):
        
        lx= left_image.shape[1]
        ly=left_image.shape[0]
        for i, bb in enumerate(left_boxes):
                cls = bb.cls.cpu().numpy()
                xywh = bb.xywh.cpu().numpy()
                # conf = bb[0][2]

                x = xywh[0][0]
                y = xywh[0][1]
                w = xywh[0][2]
                h = xywh[0][3]
        
                x1 = int((x - w / 2))      
                y1 = int((y - h / 2 ))
                x2 = int((x + w / 2))
                y2 = int((y + h / 2))
                cv2.rectangle(left_image, (x1,y1), (x2,y2), (70,255,255), 1)

                data = f"{np.round(data_to_write[i],2)}"
                cv2.putText(left_image, data , (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100,255,100), 1, cv2.LINE_AA) 
        
        path = '/home/atharav/IITBDV-main-stack/build/perception/perception/run_images_ros/camleft{0}.png'
        cv2.imwrite(path , left_image)  ## this path shld come from config file


def update_pos(msg):

        """
        For sending data to SLAM about location and yaw of car when images were captured

        """

        bot_x = msg.pose.position.x
        bot_y = msg.pose.position.y
        q = msg.pose.orientation
        car_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z) 
        bot_yaw = car_yaw
        
def process_image(image_msg):
        
        image = msg_to_cv2(image_msg)
        image = resize(image)
        return image

def msg_to_cv2(image_msg):
        
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        return image

def resize(image):

        image = cv2.resize(image, [1280, 720])
        return image

def smoothen(left = None, right = None):

        return left, right

def blur(left = None, right = None):

        return left, right

def take_img_fusion(left_image, right_image):

        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) 
        left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)

        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) 
        right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)      


