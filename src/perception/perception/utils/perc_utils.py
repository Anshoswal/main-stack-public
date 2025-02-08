import cv2
from cv_bridge import CvBridge
import numpy as np

# Import perception packages:
from perception.perception_packages.view_utils import *

# Import other dependencies
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN

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
        
#         path = package_path / 'run_images_ros/camleft{image_number}.png'
#         cv2.imwrite(path , left_image)  ## this path shld come from config file


def draw_images(left_boxes , left_image , mono, data_to_write, package_path = None):
        path = package_path / 'run_images_ros/camleft{0}.png'
        cv2.imwrite(path , left_image)  ## this path shld come from config file
        
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


                data1 = f"{np.round(data_to_write[i],2)}"
                data2 = f"{mono[i]}"
                cv2.putText(left_image, data1 , (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA) 
                cv2.putText(left_image, data2 , (x1,y1+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        
        path = '/home/jetson/IITBDV-main-stack/src/perception/perception/run_images_ros/fusion.png'
        data = f"{np.round(data_to_write[i],2)}"
        cv2.putText(left_image, data , (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100,255,100), 1, cv2.LINE_AA) 
        
        cv2.imwrite(path , left_image)  ## this path shld come from config file

def cones_xy(array_3d,labels): # array3d = cones -> x,y,z
    cones_total = len(set(labels))
    cones = np.zeros((cones_total,4)) # x, y, x-counter, y-counter
    for i in range(array_3d.shape[0]):
        cones[labels[i],:2] += array_3d[i,:2]
        cones[labels[i],2:4] += 1 # counter++
    cones[:,:2] /= cones[:,2:4]
    return cones[:,:2]

def get_pcd(self,msg):
        global lidar_height
        points = pc2.read_points(msg,skip_nans=True)
        points = np.array([[data[0]+0.59,data[1],data[2]] for data in points if (data[0] > 0 and -lidar_height+0.35 > data[2]> -lidar_height)])
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
        lidar_xy = cones_xy(points_filtered,labels)

        # self.visualize()

def search2(cam_arr, pcd_arr, r):
    
    x_points = pcd_arr[:, 0]
    y_points = pcd_arr[:, 1]
    
    indices_to_remove = []
    
    fusion_count = 0
    mono_count = 0
    for idx, point in enumerate(cam_arr):
        X_est = point[0]
        Y_est = point[1]

        squared_distances = (x_points - X_est)**2 + (y_points - Y_est)**2
        
        if squared_distances.size == 0:
            indices_to_remove.append(idx)
            continue

        min_index = np.argmin(squared_distances)

        if squared_distances[min_index] <= r**2:
            fusion_count += 1
        #     print("LIDAR CHALAAAA", idx)
            range_ = np.sqrt(x_points[min_index]**2 + y_points[min_index]**2)
            theta = np.arctan2(y_points[min_index], x_points[min_index])
            theta = theta * 180 / np.pi # Convert from radian to degree
            
            point[0] = x_points[min_index]
            point[1] = theta
            point[3] = range_

            x_points = np.delete(x_points, min_index)
            y_points = np.delete(y_points, min_index)

        else:
            mono_count += 1
        #     print("LIDAR HAGAAAA", idx)
            range_ = np.sqrt(X_est**2 + Y_est**2)
            theta = np.arctan2(Y_est, X_est)
            theta = theta * 180 / np.pi # Convert from radian to degree
            point[1] = theta
            point[3] = range_
            
            # indices_to_remove.append(idx)

    # cam_arr = np.delete(cam_arr, indices_to_remove, axis=0)
    print("NUMBER OF FUSION = ", fusion_count)
    print("NUMBER OF MONO = ", mono_count)
    return cam_arr

def check_cone_overlap(cones_list, cones_array):
       
        for new_cone in cones_list:

            if len(cones_array) > 0: 
                for cone in cones_array:
                    if (((new_cone.location.x - cone.location.x) ** 2) + ((new_cone.location.y - cone.location.y) ** 2) )**0.5 < 0.5:

                        #TODO: Use the average of the cones instead of directly eliminating, currently we are not using average

                        break
                    else:
                        cones_array.append(new_cone)
                        break
            else:
                cones_array.append(new_cone)

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


