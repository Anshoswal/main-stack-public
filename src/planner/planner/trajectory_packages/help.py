#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import time
import traceback
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import *
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose, TwistWithCovarianceStamped, Point
from visualization_msgs.msg import Marker,MarkerArray
# from rclpy.duration import Duration
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
from eufs_msgs.msg import CanState
from eufs_msgs.srv import SetCanState
from eufs_msgs.msg import ConeArrayWithCovariance
from eufs_msgs.msg import CarState
from eufs_msgs.msg import PointArray
from dv_msgs.msg import IndexedTrack, IndexedCone
from itertools import permutations, combinations
# from dv_msgs.msg import Track
from scipy.spatial.distance import cdist
import math
from trajectory.submodules_ppc.trajectory_packages import *
from trajectory.submodules_ppc.utlities import interpolate, midline_delaunay, perp_bisect, distances, unique
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
# from test_controls.skully import *

############## VARIABLES ################
PERIOD = 0.05 #20Hz
SIMULATOR_PERCEPTION = False
SIMULATOR_SLAM = True
SLAM_DISTANCE = 15
LAP_COUNT = 4
INTERPOLATION = False
PERCEPTION_DISTANCE = 20
TRACK_WIDTH = 1.5


class path_planner(Node):
    def __init__(self):

        super().__init__('main')

        # Publihsers
        self.publish_cmd = self.create_publisher(AckermannDriveStamped, '/cmd', 5)
        self.pp_waypoint = self.create_publisher(Marker, '/waypoint', 5)
        self.car_location = self.create_publisher(Marker, '/Car_location', 5)
        self.viz_cones = self.create_publisher(MarkerArray, '/viz_cones', 1)
        self.delaunay_viz = self.create_publisher(MarkerArray, '/delaunay', 1)
        self.boundary_viz = self.create_publisher(MarkerArray,'/boundary',1)
        self.midpoint_path = self.create_publisher(MarkerArray, '/waypoint_array', 1)
        self.best_point_path = self.create_publisher(MarkerArray, '/best_point_array', 1)
        self.pb_lines = self.create_publisher(MarkerArray,'/perp_bisect', 5)
        self.stopping_path = self.create_publisher(MarkerArray, '/stopping_array', 1)
        self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/track',self.get_map, 1)
        #self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/cones',self.get_map, 1)
        #self.cones_groundtruth = self.create_subscription(IndexedTrack, '/dvsim/ground_truth/cones',self.get_map,1)
        self.get_blue_cones=self.create_publisher(MarkerArray, '/cones_blue_array', 1)
        self.get_yellow_cones=self.create_publisher(MarkerArray, '/cones_yellow_array', 1)
        self.get_new_cones=self.create_publisher(MarkerArray, 'cones_new_array', 1)
        # self.cones_perception = self.create_subscription(Track, '/perception/cones',self.get_map, 1)
        #self.timer = self.create_timer(PERIOD, self.get_map)
        self.carstate_groundtruth = self.create_subscription(CarState, '/ground_truth/state',self.get_carState, 1)
        
        
        self.states = {CanState.AS_OFF: "OFF",
                       CanState.AS_READY: "READY",
                       CanState.AS_DRIVING: "DRIVING",
                       CanState.AS_EMERGENCY_BRAKE: "EMERGENCY",
                       CanState.AS_FINISHED: "FINISHED"}

        # Autonomous missions
        self.missions = {CanState.AMI_NOT_SELECTED: "NOT_SELECTED",
                         CanState.AMI_ACCELERATION: "ACCELERATION",
                         CanState.AMI_SKIDPAD: "SKIDPAD",
                         CanState.AMI_AUTOCROSS: "AUTOCROSS",
                         CanState.AMI_TRACK_DRIVE: "TRACK_DRIVE",
                         CanState.AMI_AUTONOMOUS_DEMO: "AUTONOMOUS_DEMO",
                         CanState.AMI_ADS_INSPECTION: "ADS_INSPECTION",
                         CanState.AMI_ADS_EBS: "ADS_EBS",
                         CanState.AMI_DDT_INSPECTION_A: "DDT_INSPECTION_A",
                         CanState.AMI_DDT_INSPECTION_B: "DDT_INSPECTION_B",
                         CanState.AMI_JOYSTICK: "JOYSTICK",
                         }

        # Services
        self.ebs_srv = self.create_client(Trigger, "/ros_can/ebs")
        self.reset_srv = self.create_client(Trigger, "/ros_can/reset")
        self.set_mission_cli = self.create_client(SetCanState, "/ros_can/set_mission")
        self.reset_vehicle_pos_srv = self.create_client(Trigger,
                                                             "/ros_can/reset_vehicle_pos")
        self.reset_cone_pos_srv = self.create_client(Trigger,
                                                          "/ros_can/reset_cone_pos")

        # Timers
        # self.timer = self.create_timer(PERIOD, self.control_callback)
        self.stop_sent = False
        # Misc
        self.setManualDriving()

        # Attributes
        self.blue_cones = []

        self.yellow_cones = []
        self.t_start = time.time()
        self.t_runtime = 1000
        self.t_start = time.time()
        self.track_available = False
        self.CarState_available = False
        self.t1 = time.time() - 1
        self.integral = 0
        self.error=0
        self.prev_error = 0
        self.vel_error = 0
        self.id = 0
        self.pp_id = 0
        self.id_line = 0
        self.waypoint_id = 0
        self.best_point_id = 0
        self.cube_scale=0.1
        self.pb_id = 0
        self.car_yaw = 0
        self.delay = 2000
        self.store_path = np.array([[0,0]])
        self.orange_cones_seen = False
        self.length_of_car = 1.5
        self.distance_blue = []
        self.distance_yellow = []
        self.carState = CarState()
        self.posY = 0
        self.posX = 0
        self.fov = math.radians(120)
        self.fov_radius = 10
        self.ellipse_dim = [8,4]
        self.slam_blue_cones = []
        self.slam_yellow_cones = []
        self.slam_big_orange_cones = []
        self.slam_orange_cones = []
    
    def distance(self, cone_x, cone_y):
        return math.sqrt((cone_x)**2 + (cone_y)**2)        
    def get_map(self, data):
        '''
        Store the map of all the cones.
        Track, blue cones, yellow cones:

        small_track, 320, 30
        hairpin_increasing_difficulty, 490, 490 --equal number of cones
        its_a_mess, 60, 202
        garden_light, 101, 99
        peanut, 54, 64
        boa_constrictor, 44, 51
        rectangle, 40, 52
        comp_2021, 156, 156  --equal number of cones
        '''
        # self.get_logger().info(f'Got data from simulated perception. Blue:{len(data.blue_cones)} Yellow:{len(data.yellow_cones)}')
        # self.get_logger().info(f"Got data from perception. No of cones - {len(data.track)}")

        blue_cones = []
        yellow_cones = []
        big_orange_cones = []
        orange_cones=[]
        big_orange_cones_left = []
        big_orange_cones_right = []
        slam_blue_cones = []
        slam_yellow_cones = []
        slam_big_orange_cones = []
        slam_orange_cones = []
        
        
        if SIMULATOR_PERCEPTION: # for /ground_truth/cones 
            for cone in data.blue_cones:
                if math.sqrt(cone.point.x**2+cone.point.y**2)<=PERCEPTION_DISTANCE:
                    blue_cones.append([cone.point.x, cone.point.y])
                    #4.5
            for cone in data.yellow_cones:
                if math.sqrt(cone.point.x**2+cone.point.y**2)<=PERCEPTION_DISTANCE:
                    yellow_cones.append([cone.point.x, cone.point.y])
            for cone in data.big_orange_cones:
                if math.sqrt(cone.point.x**2+cone.point.y**2)<=PERCEPTION_DISTANCE:
                    if cone.point.y > 0: ## These positions are relative to the car, so when using gloabal map, we need to convert them to relative positions
                        big_orange_cones_left.append([cone.point.x, cone.point.y])
                    else:
                        big_orange_cones_right.append([cone.point.x, cone.point.y])
                    big_orange_cones.append([cone.point.x, cone.point.y])
            for cone in data.orange_cones:
                if math.sqrt(cone.point.x**2+cone.point.y**2)<=PERCEPTION_DISTANCE:
                    orange_cones.append([cone.point.x, cone.point.y])

        #FOR SLAMMMMMMM            
        elif SIMULATOR_SLAM: # for /ground_truth/track assuming this has the same data format as ground_truth/cones
            f_tire_x = self.posX + self.length_of_car/2 * math.cos(self.car_yaw)
            f_tire_y = self.posY + self.length_of_car/2 * math.sin(self.car_yaw)
            distance_blue = []
            distance_yellow = []
            heading_vector = np.array([math.cos(self.car_yaw), math.sin(self.car_yaw)])
            #only storing cones that are seen till time t in the slam array
            #print(data)
            for cone in data.blue_cones:
                if self.is_in_slam(self.posX,self.posY,self.car_yaw,cone.point.x,cone.point.y,self.fov,self.fov_radius):
                    if [cone.point.x,cone.point.y] not in self.slam_blue_cones:
                        self.slam_blue_cones.append([cone.point.x, cone.point.y])

            for cone in data.yellow_cones:
                if self.is_in_slam(self.posX,self.posY,self.car_yaw,cone.point.x,cone.point.y,self.fov,self.fov_radius):
                    if [cone.point.x,cone.point.y] not in self.slam_yellow_cones:
                        self.slam_yellow_cones.append([cone.point.x, cone.point.y])

            for cone in data.big_orange_cones:
                if self.is_in_slam(self.posX,self.posY,self.car_yaw,cone.point.x,cone.point.y,self.fov,self.fov_radius):
                    if [cone.point.x,cone.point.y] not in self.slam_big_orange_cones:
                        self.slam_big_orange_cones.append([cone.point.x, cone.point.y])

            for cone in data.orange_cones:
                if self.is_in_slam(self.posX,self.posY,self.car_yaw,cone.point.x,cone.point.y,self.fov,self.fov_radius):
                    if [cone.point.x,cone.point.y] not in self.slam_orange_cones:
                        self.slam_orange_cones.append([cone.point.x, cone.point.y])

            #FINDING CONES IN OUR REGION OF INTEREST FROM THE SLAM CONES
            for cone in self.slam_blue_cones:
                #if math.sqrt((cone[0] - self.posX)**2+(cone[1] - self.posY)**2) <=SLAM_DISTANCE:
                
                if self.cone_in_ellipse(self.posX,self.posY,self.car_yaw,cone[0],cone[1],self.ellipse_dim[0],self.ellipse_dim[1]):
                    blue_cones.append([cone[0], cone[1],1])
                    position_vector = np.array([(cone[0]-f_tire_x),(cone[1] - f_tire_y)])
                    dot_product = np.dot(heading_vector, position_vector)
                    if dot_product>0:
                        distance_blue.append(math.sqrt((cone[0]- f_tire_x)**2 + (cone[1] - f_tire_y)**2))
                    else:
                        distance_blue.append((-1)*math.sqrt((cone[0]- f_tire_x)**2 + (cone[1] - f_tire_y)**2))

            for cone in self.slam_yellow_cones:
                #if math.sqrt((cone[0] - self.posX)**2+(cone[1] - self.posY)**2) <=SLAM_DISTANCE:
                
                if self.cone_in_ellipse(self.posX,self.posY,self.car_yaw,cone[0],cone[1],self.ellipse_dim[0],self.ellipse_dim[1]):
                    yellow_cones.append([cone[0], cone[1],0])
                    position_vector = np.array([(cone[0]-f_tire_x),(cone[1] - f_tire_y)])
                    dot_product = np.dot(heading_vector, position_vector)
                    if dot_product>0:
                        distance_yellow.append(math.sqrt((cone[0]- f_tire_x)**2 + (cone[1] - f_tire_y)**2))
                    else:
                        distance_yellow.append((-1)*math.sqrt((cone[0] - f_tire_x)**2 + (cone[1] - f_tire_y)**2))
            for cone in self.slam_big_orange_cones:
                #if math.sqrt((cone[0] - self.posX)**2+(cone[1] - self.posY)**2) <=SLAM_DISTANCE:
                
                if self.cone_in_ellipse(self.posX,self.posY,self.car_yaw,cone[0],cone[1],self.ellipse_dim[0],self.ellipse_dim[1]):
                    if cone[1] > 0: ## These positions are relative to the car, so when using gloabal map, we need to convert them to relative positions
                        big_orange_cones_left.append([cone[0], cone[1]])
                    else:
                        big_orange_cones_right.append([cone[0], cone[1]])
                    big_orange_cones.append([cone[0], cone[1]])
                    
            for cone in self.slam_orange_cones:
               
                if self.cone_in_ellipse(self.posX,self.posY,self.car_yaw,cone[0],cone[1],self.ellipse_dim[0],self.ellipse_dim[1]):    
                #if math.sqrt((cone[0] - self.posX)**2+(cone[1] - self.posY)**2) <=SLAM_DISTANCE:
                    orange_cones.append([cone[0], cone[1]])  
        
        else: # for /perception/cones
            for cone in data.track:
                if cone.color == 0:
                    blue_cones.append([(cone.location.x)*cos(cone.location.y), (cone.location.x)*sin(cone.location.y)])
                elif cone.color == 1:
                    yellow_cones.append([(cone.location.x)*cos(cone.location.y), (cone.location.x)*sin(cone.location.y)])
                elif cone.color == 2:
                    big_orange_cones.append([(cone.location.x)*cos(cone.location.y), (cone.location.x)*sin(cone.location.y)])
                elif cone.color == 3:
                    orange_cones.append([(cone.location.x)*cos(cone.location.y), (cone.location.x)*sin(cone.location.y)])
        
        self.blue_cones = np.array(blue_cones)
        self.yellow_cones = np.array(yellow_cones)
        self.big_orange_cones = np.array(big_orange_cones)
        self.orange_cones = np.array(orange_cones)
        self.big_orange_cones_left = np.array(big_orange_cones_left)
        self.big_orange_cones_right = np.array(big_orange_cones_right)
        self.distance_blue = np.array(distance_blue)       
        self.distance_yellow = np.array(distance_yellow)
        # self.cones_groundtruth.destroy()  # So that /ground/track is subscribed only once
        
        '''try:
            self.get_waypoints()
        except:
            pass'''
        self.get_waypoints()

        self.visualize_cones()
        return None
        
    def get_carState(self, data):
        # print('Car state updated')
        self.posX = data.pose.pose.position.x
        self.posY = data.pose.pose.position.y
        x,y,z,w = (data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)
        self.car_yaw = self.quaternionToYaw(x,y,z,w)
        self.carState = data
        self.CarState_available = True
        return None  
    
    def get_waypoints(self):
        '''
        Takes in and stores the Track information provided by fsds through a latched (only publishes once) ros topic. 
        '''
        # print('Generating waypoints')
        # bounds_left=evaluate_bezier(self.blue_cones,3)
        # bounds_right=evaluate_bezier(self.yellow_cones,3)
        # print(self.big_orange_cones)
        
        '''
        ~This part of code is for the stopping mechanism~
        This part only runs after the time mentioneed in 'self.delay' passes after the node is ran.
        When the big orange cones aren't visible anymore after being seen by car, this code sends signal to controller to stop the car.
        '''
        if(time.time()-self.t_start > self.delay):
            if(len(self.big_orange_cones) != 0):
                self.orange_cones_seen = True
            elif(len(self.big_orange_cones) == 0 and self.orange_cones_seen == True):
                self.send_stopping_path(self.big_orange_cones)
        f_tire_x = self.posX + self.length_of_car/2 * math.cos(self.car_yaw)
        f_tire_y = self.posY + self.length_of_car/2 * math.sin(self.car_yaw)

        
        car_yaw = 0 #Need to change this to the actual car yaw and similarly pose

        
        def compare_position(cone1):
            expected_depth=PERCEPTION_DISTANCE
            if(((cone1[0]-f_tire_x)**2+(cone1[1]-f_tire_y)**2-(expected_depth)**2)<0):
                return -1
            else:
                return 1

        
        bluecones_withcolor_unfiltered = np.array([[cone[0],cone[1],0] for cone in self.blue_cones])
        bluecones_withcolor=[]
        yellowcones_withcolor_unfiltered = np.array([[cone[0],cone[1],1] for cone in self.yellow_cones])
        yellowcones_withcolor=[]
        
        for i in range(len(bluecones_withcolor_unfiltered)):
            if compare_position(bluecones_withcolor_unfiltered[i])==-1:
                bluecones_withcolor.append(bluecones_withcolor_unfiltered[i])

        for i in range(len(yellowcones_withcolor_unfiltered)):
            if compare_position(yellowcones_withcolor_unfiltered[i])==-1:
                yellowcones_withcolor.append(yellowcones_withcolor_unfiltered[i])
        # print(bluecones_withcolor)

        bluecones_withcolor = np.array(bluecones_withcolor)
        yellowcones_withcolor = np.array(yellowcones_withcolor)
        heading_vector = [math.cos(car_yaw), math.sin(car_yaw)]
        distances_blue = [np.sqrt((bluecones_withcolor[i,0] - f_tire_x)**2 + (bluecones_withcolor[i,1] - f_tire_y)**2) for i in range(len(bluecones_withcolor))]
        distances_yellow = [np.sqrt((yellowcones_withcolor[i,0] - f_tire_x)**2 + (yellowcones_withcolor[i,1] - f_tire_y)**2) for i in range(len(yellowcones_withcolor))]
        
        sorted_dist_blue = np.argsort(self.distance_blue)
        sorted_dist_yellow = np.argsort(self.distance_yellow)
        
        sorted_blue_cones = self.blue_cones[sorted_dist_blue]
        sorted_yellow_cones = self.yellow_cones[sorted_dist_yellow]
        # print("Here it is")
        # print(sorted_blue_cones)
        # blue_boundary=np.array(2,3)
        # yellow_boundary=np.zeros((len(sorted_yellow_cones)/2))
        # blue_boundary=list(sorted_blue_cones)
        # yellow_boundary=list(sorted_yellow_cones)
        blue_boundary=[]
        yellow_boundary=[]
        # print(blue_boundary)  
        # for k in range (0,len(sorted_blue_cones)/2):
        #     blue_boundary[i]=[0,0,0,0]
        #     yellow_boundary[i]=[0,0,0,0]
        # print(blue_boundary)   
        

        #To get boundary of the cones
        #Creates the blue_boundary array with constaints of check_track
        for j in range(0,len(sorted_blue_cones)-1):
            if self.check_track(sorted_blue_cones[j-1,0],sorted_blue_cones[j-1,1],sorted_blue_cones[j,0],sorted_blue_cones[j,1],sorted_blue_cones[j+1,0],sorted_blue_cones[j+1,1],j):
                blue_boundary.append([sorted_blue_cones[j,0],sorted_blue_cones[j,1],sorted_blue_cones[j+1,0],sorted_blue_cones[j+1,1]])
            else:
                break

        for j in range(0,len(sorted_yellow_cones)-1):
            if self.check_track(sorted_yellow_cones[j-1,0],sorted_yellow_cones[j-1,1],sorted_yellow_cones[j,0],sorted_yellow_cones[j,1],sorted_yellow_cones[j+1,0],sorted_yellow_cones[j+1,1],j):
                yellow_boundary.append([sorted_yellow_cones[j,0],sorted_yellow_cones[j,1],sorted_yellow_cones[j+1,0],sorted_yellow_cones[j+1,1]])
            else:
                break
        
        self.visualize_boundary(blue_boundary,0)
        self.visualize_boundary(yellow_boundary,1)
        # print("blue_boundary",blue_boundary)
        # print("yellow_boundary",yellow_boundary)
        old_detected_blue_cones = MarkerArray()
        old_detected_yellow_cones = MarkerArray()
        #print(yellowcones_withcolor)
        number_blue=0
        number_yellow=0
        
        for cone in self.blue_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
    
            # if color==0:
            color_cone_r = 1.0
            color_cone_g = 1.0
            color_cone_b = 1.0
        
            
            marker.id = self.waypoint_id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'map'
            old_detected_blue_cones.markers.append(marker)
            self.waypoint_id += 1
            number_blue+=1
        self.get_blue_cones.publish(old_detected_blue_cones)
        # print("Hello")
        for cone in self.yellow_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
    
            # if color==0:
            color_cone_r = 1.0
            color_cone_g = 1.0
            color_cone_b = 1.0
        
            
            marker.id = self.waypoint_id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'map'
            old_detected_yellow_cones.markers.append(marker)
            self.waypoint_id += 1
            number_yellow+=1
        self.get_yellow_cones.publish(old_detected_yellow_cones)
        
        try:
            x_mid, y_mid, line_list , line_length_list = self.midline_delaunay(self.blue_cones, self.yellow_cones)
            x_mid = np.array(x_mid)
            y_mid = np.array(y_mid)
            xy_mid = np.column_stack((x_mid,y_mid))
            xy_mid_send = np.copy(xy_mid)
            #####Modify to only have the points in front of the car
            xy_mid_send = self.filter_points_by_distance(xy_mid_send ,(self.posX,self.posY))
            possible_paths = self.evaluate_possible_paths(xy_mid_send , [self.posX , self.posY] )#possible paths is currently a list
            
            print("length of possible paths",len(possible_paths))
            best_path,min_path_cost = self.choose_best_path(possible_paths,line_length_list,xy_mid_send)
            
            print("best path",best_path)
            print("xy_mid try",xy_mid)
            
            
        except:
            #trying to get pseudo way points by the perp bisector method
            x_mid, y_mid, mid_point_cones_array, our_points = self.perp_bisect(self.blue_cones,self.yellow_cones, TRACK_WIDTH)
            self.visualize_perp(mid_point_cones_array,our_points)
            xy_mid = np.column_stack((x_mid,y_mid))
        #self.midline_delaunay(x_mid,y_mid)
    
    
        #####Modify to only have the points in front of the car
        
    
        '''
        This below part of code is used to interpolate points
        '''
        
        if INTERPOLATION and len(np.unique(x_mid))>1:
            distances_from_midpoints = distances(f_tire_x, f_tire_y, x_mid=best_path[:][0], y_mid=best_path[:][1])
            xy_mid_send = interpolate(x_mid=best_path[:][0], y_mid=best_path[:][1], distances=distances_from_midpoints)
        # self.get_logger().info(f'Xymid:{xy_mid}')
        try:
            self.visualize_line(line_list)
        except:
            pass 
        # The points aren't ordered here, can't directly interpolate
        # print(xy_mid)


        # xy_mid_interpolated = evaluate_bezier(xy_mid, 5)
        print("xy_mid try:",xy_mid)
        try:
            self.send_midpoint_path(xy_mid=best_path)
        except:
            self.send_midpoint_path(xy_mid=xy_mid)
        # self.track_available = True

        return None
        # self.get_blue_cones.publish(old_detected_blue_cones)
        # self.get_yellow_cones.publish(old_detected_yellow_cones)


   

    def filter_points_by_distance(self , points, car_position, threshold=2):
        car_x, car_y = car_position
        # Compute distances using vectorized operations
        distances = np.sqrt((points[:, 0] - car_x)**2 + (points[:, 1] - car_y)**2)
        # Filter points whose distance is greater than the threshold
        filtered_points = points[distances > threshold]
        return filtered_points





    '''
    def evaluate_possible_paths(self, xy_mid, starting_point, n = 3):
            possible_paths = []
    
            # Get all combinations of `n` midpoints
            midpoint_combinations = combinations(xy_mid, n)
            #print(midpoint_combinations
            
            for combination in midpoint_combinations:
                # Sort the midpoints in the combination by their distance from the starting point
                
                #combination = (starting_point,)  + combination
                combination = np.array(combination)
                sorted_combination = [np.array(starting_point)]
                current_point = starting_point
                while len(combination) > 0:
                    # Compute Euclidean distances
                    distances = np.linalg.norm(combination - current_point, axis=1)
                    # Find the index of the closest point
                    closest_index = np.argmin(distances)
                    # Add the closest point to the sorted list
                    closest_point = combination[closest_index]
                    sorted_combination.append(closest_point)
                    # Remove the closest point from the list of points
                    combination = np.delete(combination, closest_index, axis=0)
                    # Update the current point
                    current_point = closest_point
                possible_paths.append(sorted_combination)
                #print(combination)
            return possible_paths
                
    def choose_best_path(self,possible_paths,line_length_list, xy_mid ,standard_width=3, angle_weight=1.4, edge_weight=0.4, k_path_weight = 10, expected_path_length_avg = 1):
        min_path_cost = float('inf')
        best_path = None
        for path in possible_paths:
            path_cost = 0

            # Calculate costs for the path
            for i in range(len(path) - 2):
                angle_cost = self.calculate_angle_cost(path[i], path[i + 1], path[i + 2])
                path_cost += angle_weight * angle_cost
                #print('angle_cost',i,angle_cost)
            e=0
            
            for midpoint in path[1:]:
                edge_cost = self.calculate_edge_length_cost(midpoint,line_length_list, xy_mid ,standard_width)
                path_cost += edge_weight * edge_cost
                e+=1
                #print('edge_cost',e,edge_cost)
            
            path_len_cost = abs(self.path_length_avg(path[1:]) - expected_path_length_avg)
            path_cost += path_len_cost * k_path_weight
            # Update best path if this path has a lower cost
            if path_cost < min_path_cost:
                min_path_cost = path_cost
                best_path = path
        #print('min',min_path_cost)
        self.best_points(best_path)
        return best_path,min_path_cost
        

    #COST FUNCTIONSSS
    ##################
    def calculate_angle_cost(self,midpoint1, midpoint2, midpoint3):
        
        #Calculate the cost based on the angle change between the lines formed by three consecutive midpoints.
        
        #try:
            #vector1 = midpoint2.coordinates - midpoint1.coordinates
            #vector2 = midpoint3.coordinates - midpoint2.coordinates
        #except:
         #   vector1 = midpoint2.coordinates - [0,0]
         #   vector2 = midpoint3.coordinates - midpoint2.coordinates
        vector1 = midpoint2 - midpoint1
        vector2 = midpoint3 - midpoint2
        
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        cosine_angle = np.dot(unit_vector1, unit_vector2)
        angle_cost = 1 - cosine_angle  # Closer to 0 means smaller angle deviation
        return angle_cost

    def calculate_edge_length_cost(self,midpoint,line_length_list,  xy_mid ,standard_width=3.0):
        
        #Calculate the cost based on the deviation of edge length from a standard track width.
        
        
        ####
        edge_length = None 
        for i,point in enumerate(xy_mid):
            if point.all() == midpoint.all():
                edge_length = line_length_list[i]
        if edge_length is None:
            raise ValueError(f"Midpoint {midpoint} not found in the list of midpoints.")

        # Calculate the edge cost
        edge_cost = abs(edge_length - standard_width)
        return edge_cost


    def path_length_avg(self,possible_path):
        #print(possible_path)
        possible_path_coordinates = []
        for way_point in possible_path:
            
            way_point = np.array(way_point)
            possible_path_coordinates.append(way_point)
        
        path_length = sum(np.linalg.norm(possible_path_coordinates[i]-possible_path_coordinates[i+1]) for i in range(len(possible_path_coordinates)-1))
        #print(path_length)
        return path_length/len(possible_path)
    ###########
        '''




#########
    def evaluate_possible_paths(self, xy_mid, starting_point, n=4):
        """
        Generates all possible paths using combinations and sorts the midpoints by distance to minimize latency.
        """
        possible_paths = []
        starting_point = np.array(starting_point)

        # Generate all combinations of n midpoints
        midpoint_combinations = list(combinations(xy_mid, n))

        for combination in midpoint_combinations:
            combination = np.array(combination)

            # Compute distances using SciPy's cdist for vectorized distance calculations
            current_point = starting_point
            sorted_combination = [starting_point]

            while combination.shape[0] > 0:
                distances = cdist([current_point], combination, metric="euclidean").flatten()
                closest_index = np.argmin(distances)
                closest_point = combination[closest_index]

                sorted_combination.append(closest_point)
                combination = np.delete(combination, closest_index, axis=0)
                current_point = closest_point

            possible_paths.append(np.array(sorted_combination))
        
        return possible_paths


    def choose_best_path(self, possible_paths, line_length_list, xy_mid, standard_width=3, angle_weight=1.4, edge_weight=0.4, k_path_weight=10, expected_path_length_avg=1):
        """
        Selects the best path based on cost function optimization.
        """
        min_path_cost = float('inf')
        best_path = None

        # Preprocess edge lengths into a dictionary for O(1) lookup
        edge_length_dict = {tuple(midpoint): length for midpoint, length in zip(xy_mid, line_length_list)}

        for path in possible_paths:
            path_cost = 0

            # Vectorized angle cost calculation
            vectors = np.diff(path, axis=0)
            unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            cosine_angles = np.einsum('ij,ij->i', unit_vectors[:-1], unit_vectors[1:])
            angle_costs = 1 - cosine_angles  # Angle cost formula
            path_cost += angle_weight * np.sum(angle_costs)

            # Vectorized edge cost calculation
            edge_costs = [
                abs(edge_length_dict[tuple(midpoint)] - standard_width)
                for midpoint in path[1:]
            ]
            path_cost += edge_weight * sum(edge_costs)

            # Path length cost
            path_len_cost = abs(self.path_length_avg(path[1:]) - expected_path_length_avg)
            path_cost += k_path_weight * path_len_cost

            # Check for the minimum path cost
            if path_cost < min_path_cost:
                min_path_cost = path_cost
                best_path = path

        return best_path, min_path_cost

    def path_length_avg(self, possible_path):
        """
        Vectorized calculation of average path length.
        """
        differences = np.diff(possible_path, axis=0)
        segment_lengths = np.linalg.norm(differences, axis=1)
        return np.sum(segment_lengths) / len(segment_lengths)

    # Example cost functions remain the same
    def calculate_angle_cost(self, midpoint1, midpoint2, midpoint3):
        vector1 = midpoint2 - midpoint1
        vector2 = midpoint3 - midpoint2
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return 1 - cosine_angle

    def calculate_edge_length_cost(self, midpoint, line_length_list, xy_mid, standard_width=3.0):
        return abs(line_length_list[xy_mid.index(midpoint)] - standard_width)



    def midline_delaunay(self, blue_cones, yellow_cones):
        new_current_detected_cones = np.append(blue_cones,yellow_cones, axis = 0)
        new_current_detected_cones = np.array(new_current_detected_cones)
        #print(bluecones_withcolor)
        #print(new_current_detected_cones)
        print("heliiii")
        x_mid, y_mid = [], []
        triangulation = Delaunay(new_current_detected_cones[:,:2])
        index=triangulation.simplices
        new_list3 = [[[0,0,0], [0,0,0], [0,0,0]] for _ in range(len(index))]

        for i in range(len(index)):
            new_list3[i][0] = new_current_detected_cones[index[i][0]]
            new_list3[i][1] = new_current_detected_cones[index[i][1]]
            new_list3[i][2] = new_current_detected_cones[index[i][2]]

        print("hehe")
        #vertices= triangulation.points[c]
        print("SOB")
        # p=0
        q = 0
        # cntr=0
    
        print("NOOOOOOOOOOOo")
        

        line_list=[]
        #print("new list", new_list3)
        line_length_list = []
        print("Yooo")
            
        for triangle in new_list3:
            for i in [0,1]:
                for j in range(i+1,3):
                    point_1x=triangle[i][0]
                    point_2x=triangle[j][0]
                    point_1y=triangle[i][1]
                    point_2y=triangle[j][1]   
                    line_length = math.sqrt((point_1x-point_2x)**2 + (point_1y-point_2y)**2)                
                    if triangle[i][2]!=triangle[j][2]:
                        #2.9 and 4
                        line_mid_x = (triangle[i][0]+triangle[j][0])/2
                        line_mid_y = (triangle[i][1]+triangle[j][1])/2
                        line_list.append([triangle[i][0],triangle[i][1],triangle[j][0],triangle[j][1]])
                        line_length_list.append(line_length)
                        x_mid.append(line_mid_x)
                        y_mid.append(line_mid_y)
        

        return x_mid, y_mid, line_list , line_length_list



    def perp_bisect(self, blue_cones, yellow_cones, TRACK_WIDTH=1.5):
        x_mid, y_mid = [], []
        #trying to get pseudo way points by the perp bisector method
        mid_point_cones_array = np.array([[0,0]])
        if len(blue_cones)>=2:
            perp_cones = blue_cones
            print('case1')
            print(f'bluecones:{blue_cones}yellowcones:{yellow_cones}')
        elif len(yellow_cones)>=2:
            perp_cones = yellow_cones
            print('case2')
            print(f'bluecones:{blue_cones}yellowcones:{yellow_cones}')
        
        dist_cones = []
        for cone in perp_cones:
            dist_cone = math.sqrt((cone[0]**2)+(cone[1]**2))#distance of cones wrt car in car's frame
            dist_cones.append(dist_cone)
        dist_cones = np.array(dist_cones)
        sorted_indices = np.argsort(dist_cones)#sorting wrt the distance
        sorted_perp_cones = [perp_cones[sorted_i] for sorted_i in sorted_indices]#gives us the sorted cones array which will be used for perp bisector so we can take their pairs
        our_points = []
        for i in range(len(sorted_perp_cones)-1):
            mid_point_cones = np.array([(sorted_perp_cones[i][0]+sorted_perp_cones[i+1][0])/2,(sorted_perp_cones[i][1]+sorted_perp_cones[i+1][1])/2])
            mid_point_cones_array = np.append(mid_point_cones_array, [mid_point_cones], axis=0)
            #gives us array of mid points of pair of cones that we take
            #unit direction vector

            magnitude = math.sqrt((sorted_perp_cones[i][0]-sorted_perp_cones[i+1][0])**2 + (sorted_perp_cones[i][1]-sorted_perp_cones[i+1][1])**2)

            unit_perpendicular_vector = np.array([(sorted_perp_cones[i][1]-sorted_perp_cones[i+1][1])/magnitude,(sorted_perp_cones[i+1][0]-sorted_perp_cones[i][0])/magnitude])

            displacement_vector = np.array([unit_perpendicular_vector[0]*TRACK_WIDTH,unit_perpendicular_vector[1]*TRACK_WIDTH])
            #1.5 is approximate half width of road

            possible_waypoints = np.array([[mid_point_cones[0]+displacement_vector[0],mid_point_cones[1]+displacement_vector[1]],[mid_point_cones[0]-displacement_vector[0],mid_point_cones[1]-displacement_vector[1]]])
            dist_from_car = np.array([math.sqrt(possible_waypoints[0][0]**2+possible_waypoints[0][1]**2),math.sqrt(possible_waypoints[1][0]**2+possible_waypoints[1][1]**2)])
            our_point_index = np.argmin(dist_from_car)
            #basically after taking perpendicular bisector we will get two points on that line with the same dist from mid point of cones and we select the point closest to car as it will lie inside the road

            our_point = possible_waypoints[our_point_index]
            our_points.append(our_point)

            print(f'perp_working, waypoint{our_point}')
            x_mid.append(our_point[0])
            y_mid.append(our_point[1])
        mid_point_cones_array = mid_point_cones_array[1:]
        

        return x_mid, y_mid, mid_point_cones_array, our_points

    def check_track(self, x0,y0,x1,y1,x2,y2,index):#Checks all the line segments of our boundary with angle and length constraints to avoid abnormal boundaries
        #Calculate the length of a line given its start and end points.
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        #Calculate the angle between two consecutive lines.
        # Vectors of the two lines
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x1 - x0, y1 - y0])
    
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
    
        # Avoid division by zero
        if v1_norm == 0 or v2_norm == 0:
            return 0

        # Compute cosine of the angle using the dot product
        dot_product = np.dot(v1, v2)
        cos_theta = dot_product / (v1_norm * v2_norm)
    
        # Clip cos_theta to avoid numerical errors outside the range of arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
        # Return the angle in degrees
        angle = np.degrees(np.arccos(cos_theta))
        if index == 0:
            return True
        elif index>0 and length<3 and angle<30:
            return True
        else:
            return False

    


    

    def is_in_slam(self,car_x, car_y, car_theta, cone_x, cone_y, fov_rad, radius):#Checks whether a cone is in the car's field of view which is an arc
        # Calculate the distance between the car and the cone
        dx = cone_x - car_x
        dy = cone_y - car_y
        distance = math.sqrt(dx**2 + dy**2)
    
        # Check if the cone is within the radius
        if distance > radius:
            return False
    
        # Calculate the angle between the car's orientation and the cone's position
        angle_to_cone = math.atan2(dy, dx)
    
        # Normalize angle differences to be between -pi and pi
        angle_diff = (angle_to_cone - car_theta + math.pi) % (2 * math.pi) - math.pi
    
        # Check if the cone is within the FOV
        return -fov_rad / 2 <= angle_diff <= fov_rad / 2
    
    def cone_in_ellipse(self,car_x, car_y, car_theta, cone_x, cone_y,a,b):#Checks among the cones if they lie in an ellipse around the car at time t
        # Translate cone position relative to car
        translated_x = cone_x - car_x
        translated_y = cone_y - car_y

        # Rotate coordinates to align with the car's orientation
        rotated_x = (translated_x * np.cos(-car_theta) - 
                    translated_y * np.sin(-car_theta))
        rotated_y = (translated_x * np.sin(-car_theta) + 
                    translated_y * np.cos(-car_theta))
    
        # Check if the point is inside the ellipse
        return (rotated_x**2 / a**2) + (rotated_y**2 / b**2) <= 1 and rotated_x > 0

    def visualize_perp(self, mid_point_cones_array, perp_bisector_array):
        perp_bisect = MarkerArray()
        for i in range (len(mid_point_cones_array)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = self.pb_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  # Line width
            # Set the points of the line
            point_1 = Point()
            point_1.x = mid_point_cones_array[i][0]
            point_1.y = mid_point_cones_array[i][1]
            point_1.z = 0.0

            point_2 = Point()
            point_2.x = perp_bisector_array[i][0]
            point_2.y = perp_bisector_array[i][1]
            point_2.z = 0.0
            marker.points = [point_1,point_2]
            # print(marker.points)
            # Set the color (red in this case)
            marker.color.r = 1.0
            marker.color.a = 1.0  # Alpha value
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 5000000
            marker.lifetime = Duration_of_marker  # Permanent marker
            self.pb_id += 1
            perp_bisect.markers.append(marker)
        self.pb_lines.publish(perp_bisect)


    def best_points(self, best_path):
        best_points = MarkerArray()
        print("best",best_path)
        for i in range (len(best_path)-1):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = self.best_point_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  # Line width
            # Set the points of the line
            point_1 = Point()
            point_1.x = float(best_path[i][0])
            point_1.y = float(best_path[i][1])
            point_1.z = 0.0

            point_2 = Point()
            point_2.x = float(best_path[i+1][0])
            point_2.y = float(best_path[i+1][1])
            point_2.z = 0.0
            marker.points = [point_1,point_2]
            # print(marker.points)
            # Set the color (red in this case)
            marker.color.r = 1.0
            marker.color.a = 1.0  # Alpha value
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 5000000
            marker.lifetime = Duration_of_marker  # Permanent marker
            self.best_point_id += 1
            best_points.markers.append(marker)
            
        self.best_point_path.publish(best_points)

    def send_midpoint_path(self, xy_mid):

        waypoints_msg = MarkerArray()
        
        for x,y in xy_mid:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
    
            color_cone_r = 0.0
            color_cone_g = 1.0
            color_cone_b = 0.0
            
            
            marker.id = self.waypoint_id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(x) #x
            marker.pose.position.y = float(y) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'map'
            waypoints_msg.markers.append(marker)
            self.waypoint_id += 1

        self.midpoint_path.publish(waypoints_msg)

    def send_stopping_path(self, cones):
        #The function's content doesn't matter because it should just publish to the topic so that controller can understand it has to stop the car
        stoppoints_msg = MarkerArray()
        
        for x,y in cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
    
            
            color_cone_r = 1.0
            color_cone_g = 1.0
            color_cone_b = 1.0
        
            
            marker.id = self.waypoint_id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(x) #x
            marker.pose.position.y = float(y) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'map'
            stoppoints_msg.markers.append(marker)
            self.waypoint_id += 1
        self.stopping_path.publish(stoppoints_msg)
  
    
    '''
    def timer_callback(self):
        control_msg = AckermannDriveStamped()
        # control_msg.header.stamp = Node.get_clock(self).now().to_msg()
        # control_msg.header.
        control_msg.drive.steering_angle = 0.0
        control_msg.drive.steering_angle_velocity = 0.0
        control_msg.drive.speed = 0.0
        control_msg.drive.acceleration = 0.5
        control_msg.drive.jerk = 0.0
        self.publish_cmd.publish(control_msg)
        return None
    
    def control_callback(self):
       
       # Check track availablility, return if not present
        if (self.track_available and self.CarState_available) == False:
            self.t_start = time.time()
            self.get_logger().info(f'Track Available:{self.track_available} CarState available:{self.CarState_available}')
            return

        # Run Node for limited time 
        if time.time() < self.t_start + self.t_runtime :
            # print('Enter Control loop')

            pos_x = self.carState.pose.pose.position.x
            pos_y = self.carState.pose.pose.position.y
            self.store_path = np.append(self.store_path, [[pos_x,pos_y]], axis = 0)
            q = self.carState.pose.pose.orientation
            v_curr=np.sqrt(self.carState.twist.twist.linear.x**2 + self.carState.twist.twist.linear.y**2)
            

            car_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

            kp =  0.4
            ki =  0.001
            kd =  0.02
            dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
            self.t1 = time.time()
            
            closest_waypoint_index=np.argmin((pos_x-self.x)**2+(pos_y-self.y)**2)
            
            [throttle,brake,self.integral,self.vel_error,diffn ] = vel_controller2(kp=kp, ki=ki, kd=kd,
                                                        v_curr=v_curr, v_ref=8,
                                                        dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)
            # print('close_index',closest_waypoint_index)
            # print('no. of midpoints',self.midpoints.shape)
            # print('paired',self.paired_indexes)

            [steer_pp, x_p, y_p] = pure_pursuit(x=self.x, y=self.y, vf=v_curr, pos_x=0, pos_y=0, 
                                                veh_head=0,pos=closest_waypoint_index, K = 1, L = 1.6)

            # print('waypoint',x_p,y_p)
            # self.visualize_cones()
            self.visualize_pp_waypoint(x_pp = x_p,y_pp = y_p)
            self.visualize_car(x_coord = pos_x, y_coord = pos_y)
            #print('following',x_p,y_p)
            #print('position',pos_x,pos_y)
            #print('steer',steer_pp,'yaw',car_yaw)

                # carControlsmsg.throttle = throttle
                # carControlsmsg.brake = brake
                # carControlsmsg.steering = steer_pp

                # carControls.publish(carControlsmsg)

            # print(f'Steer:{steer_pp}, Accn:{throttle - brake}, Car Yaw:{car_yaw}, Car Pos:{pos_x, pos_y}, PP point:{x_p, y_p}')
            control_msg = AckermannDriveStamped()
            # control_msg.header.stamp = Node.get_clock(self).now().to_msg()
            # control_msg.header.
            print(steer_pp)
            control_msg.drive.steering_angle = float(steer_pp)
            control_msg.drive.acceleration = float(throttle - brake)
            # control_msg.drive.acceleration = 0.05
            self.publish_cmd.publish(control_msg)

            self.get_logger().info(f'Speed:{v_curr:.4f} Accn:{float(throttle - brake):.4f} Steering:{float(-steer_pp):.4f} Time:{time.time() - self.t_start:.4f} x_p:{x_p} y_p:{y_p}')

        else:
            raise SystemExit
            
        return None
    '''
    def quaternionToYaw(self,x,y,z,w):
        yaw = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
        return yaw
    def visualize_pp_waypoint(self, x_pp, y_pp):
        data = Pose()
        data.position.x = float(x_pp)
        data.position.y = float(y_pp)
        pure_pursuit_waypoint_msg = Marker()
        pure_pursuit_waypoint_msg.header.frame_id = 'map'
        pure_pursuit_waypoint_msg.ns = "Way_ppoint"
        pure_pursuit_waypoint_msg.id = self.pp_id
        self.pp_id += 1
        pure_pursuit_waypoint_msg.type = 1
        pure_pursuit_waypoint_msg.action = 0
        pure_pursuit_waypoint_msg.pose = data
        pure_pursuit_waypoint_msg.scale.x = self.cube_scale
        pure_pursuit_waypoint_msg.scale.y = self.cube_scale
        pure_pursuit_waypoint_msg.scale.z = self.cube_scale
        pure_pursuit_waypoint_msg.color.r = 0.0
        pure_pursuit_waypoint_msg.color.g = 256.0
        pure_pursuit_waypoint_msg.color.b = 256.0
        pure_pursuit_waypoint_msg.color.a = 1.0
        Duration_of_marker = Duration()
        Duration_of_marker.sec = 0
        Duration_of_marker.nanosec = 100000000  #0.1 seconds
        pure_pursuit_waypoint_msg.lifetime = Duration_of_marker
        self.pp_waypoint.publish(pure_pursuit_waypoint_msg)
    
    def visualize_car(self, x_coord, y_coord):
        data = Pose()
        data.position.x = float(x_coord)
        data.position.y = float(y_coord)
        car_location_msg = Marker()
        car_location_msg.header.frame_id = 'map'
        car_location_msg.ns = "Car's Location"
        car_location_msg.id = self.pp_id
        self.pp_id += 1
        car_location_msg.type = 1
        car_location_msg.action = 0
        car_location_msg.pose = data
        car_location_msg.scale.x = self.cube_scale
        car_location_msg.scale.y = self.cube_scale
        car_location_msg.scale.z = self.cube_scale
        car_location_msg.color.r = 256.0
        car_location_msg.color.g = 256.0
        car_location_msg.color.b = 256.0
        car_location_msg.color.a = 1.0
        Duration_of_marker = Duration()
        Duration_of_marker.sec = 0
        Duration_of_marker.nanosec = 50000000 # .1 seconds
        car_location_msg.lifetime = Duration_of_marker
        self.car_location.publish(car_location_msg)

    def visualize_cones(self):
        
        # print('Viz cones')
        all_cones = MarkerArray()
        
        for cone in self.blue_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
      
            # if color==0:
            color_cone_r = 0.0
            color_cone_g = 0.0
            color_cone_b = 1.0
            # if color==1:
            #         color_cone_r = 1
            #         color_cone_g = 1
            #         color_cone_b = 0
            # if color==2:
            #         color_cone_r = 1
            #         color_cone_g = 0.220
            #         color_cone_b = 0 
            # if color==3:
            #         color_cone_r = 1
            #         color_cone_g = 0.6420
            #         color_cone_b = 0.5
            # if color==4:
            #         color_cone_r = 1
            #         color_cone_g = 1 
            #         color_cone_b = 1
            
            marker.id = self.id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'map'
            all_cones.markers.append(marker)
            self.id += 1
        for cone in self.yellow_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
      
            # if color==0:
            color_cone_r = 1.0
            color_cone_g = 1.0
            color_cone_b = 0.0
            # if color==1:
            #         color_cone_r = 1
            #         color_cone_g = 1
            #         color_cone_b = 0
            # if color==2:
            #         color_cone_r = 1
            #         color_cone_g = 0.220
            #         color_cone_b = 0 
            # if color==3:
            #         color_cone_r = 1
            #         color_cone_g = 0.6420
            #         color_cone_b = 0.5
            # if color==4:
            #         color_cone_r = 1
            #         color_cone_g = 1 
            #         color_cone_b = 1
            
            marker.id = self.id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'map'
            all_cones.markers.append(marker)
            self.id += 1

        # print('Publishing cones on /viz_cones')

        # rate = rclpy.Rate(10)

        
            
        self.viz_cones.publish(all_cones)
            # time.sleep(0.05)

    def visualize_line(self, line_list):

        markerArrayMsg = MarkerArray()
        for pair_of_points in line_list:
            # print(pair_of_points)
            x1,y1,x2,y2 = pair_of_points
            for i in range(0,2): 
                for j in range(i+1,3):    
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.id = self.id_line
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.05  # Line width

                    # Set the points of the line
                    point_1 = Point()
                    point_1.x = x1
                    point_1.y = y1
                    point_1.z = 0.0

                    point_2 = Point()
                    point_2.x = x2
                    point_2.y = y2
                    point_2.z = 0.0
                    marker.points = [point_1,
                                     point_2
                                    ]
                    # print(marker.points)
                    # Set the color (red in this case)
                    marker.color.r = 1.0
                    marker.color.a = 1.0  # Alpha value
                    Duration_of_marker = Duration()
                    Duration_of_marker.sec = 0
                    Duration_of_marker.nanosec = 5000000
                    marker.lifetime = Duration_of_marker  # Permanent marker
                    markerArrayMsg.markers.append(marker)
                    self.id_line += 1
        # rate = rospy.Rate(10)
        # print('hi')
        # print(markerArrayMsg)
        # cones_viz =  rospy.Publisher('/track_lines', MarkerArray, queue_size=10)
        # while not rospy.is_shutdown():
        # self.get_logger().info(f"finsihed viz lines")
        self.delaunay_viz.publish(markerArrayMsg)

    def visualize_boundary(self, line_list,color):
        markerArrayMsg = MarkerArray()
        for pair_of_points in line_list:
            # print(pair_of_points)
            x1,y1,x2,y2 = pair_of_points
            for i in range(0,2): 
                for j in range(i+1,3):    
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.id = self.id_line
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.05  # Line width

                    # Set the points of the line
                    point_1 = Point()
                    point_1.x = x1
                    point_1.y = y1
                    point_1.z = 0.0

                    point_2 = Point()
                    point_2.x = x2
                    point_2.y = y2
                    point_2.z = 0.0
                    marker.points = [point_1,
                                     point_2
                                    ]
                    # print(marker.points)
                    # Set the color (red in this case)
                    marker.color.b = 1.0                        
                    if color==1:
                        marker.color.g = 1.0
                        marker.color.r = 1.0
                        marker.color.b = 0.0
                    marker.color.a = 1.0  # Alpha value
                    Duration_of_marker = Duration()
                    Duration_of_marker.sec = 0
                    Duration_of_marker.nanosec = 5000000
                    marker.lifetime = Duration_of_marker  # Permanent marker
                    markerArrayMsg.markers.append(marker)
                    self.id_line += 1
        # rate = rospy.Rate(10)
        # print('hi')
        # print(markerArrayMsg)
        # cones_viz =  rospy.Publisher('/track_lines', MarkerArray, queue_size=10)
        # while not rospy.is_shutdown():
        # self.get_logger().info(f"finsihed viz lines")
        self.boundary_viz.publish(markerArrayMsg)

    def sendRequest(self, mission_ami_state):
        """Sends a mission request to the simulated ros_can
        The mission request is of message type eufs_msgs/srv/SetCanState
        where only the ami_state field is used.
        """
        if self.set_mission_cli.wait_for_service(timeout_sec=1):
            request = SetCanState.Request()
            request.ami_state = mission_ami_state
            result = self.set_mission_cli.call_async(request)
            # self.node.get_logger().debug("Mission request sent successfully")
            # self.node.get_logger().debug(result)  
        else:
            # self.node.get_logger().warn(
            #     "/ros_can/set_mission service is not available")
            self.get_logger().warn(
                "/ros_can/set_mission service is not available")

    def setMission(self, mission):
        """Requests ros_can to set mission"""
        # mission = self._widget.findChild(
        #     QComboBox, "MissionSelectMenu").currentText()

        # self.node.get_logger().debug(
        #     "Sending mission request for " + str(mission))

        # create message to be sent
        mission_msg = CanState()

        # find enumerated mission and set
        for enum, mission_name in self.missions.items():
            if mission_name == mission:
                mission_msg.ami_state = enum
                break
        # mission_msg.ami_state = CanState.AMI_SKIDPAD
        self.sendRequest(mission_msg.ami_state)

    def setManualDriving(self):
        self.get_logger().debug("Sending manual mission request")
        mission_msg = CanState()
        mission_msg.ami_state = CanState.AMI_MANUAL
        self.sendRequest(mission_msg.ami_state)

    def resetState(self):
        """Requests state_machine reset"""
        self.node.get_logger().debug("Requesting state_machine reset")

        if self.reset_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.reset_srv.call_async(request)
            self.node.get_logger().debug("state reset successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/reset service is not available")

    def resetVehiclePos(self):
        """Requests race car model position reset"""
        self.node.get_logger().debug(
            "Requesting race_car_model position reset")

        if self.reset_vehicle_pos_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.reset_vehicle_pos_srv.call_async(request)
            self.node.get_logger().debug("Vehicle position reset successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/reset_vehicle_pos service is not available")

    def resetConePos(self):
        """Requests gazebo_cone_ground_truth to reset cone position"""
        self.node.get_logger().debug(
            "Requesting gazebo_cone_ground_truth cone position reset")

        if self.reset_cone_pos_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.reset_cone_pos_srv.call_async(request)
            self.node.get_logger().debug("Cone position reset successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/reset_cone_pos service is not available")

    def resetSim(self):
        """Requests state machine, vehicle position and cone position reset"""
        self.node.get_logger().debug("Requesting Simulation Reset")

        # Reset State Machine
        self.resetState()

        # Reset Vehicle Position
        self.resetVehiclePos()

        # Reset Cone Position
        self.resetConePos()

    def requestEBS(self):
        """Requests ros_can to go into EMERGENCY_BRAKE state"""
        self.node.get_logger().debug("Requesting EBS")

        if self.ebs_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.ebs_srv.call_async(request)
            self.node.get_logger().debug("EBS successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/ebs service is not available")

    def stateCallback(self, msg):
        """Reads the robot state from the message
        and displays it within the GUI

        Args:
            msg (eufs_msgs/CanState): state of race car
        """
        if msg.ami_state == CanState.AMI_MANUAL:
            self._widget.findChild(QLabel, "StateDisplay").setText(
                "Manual Driving")
            self._widget.findChild(QLabel, "MissionDisplay").setText(
                "MANUAL")
        else:
            self._widget.findChild(QLabel, "StateDisplay").setText(
                self.states[msg.as_state])
            self._widget.findChild(QLabel, "MissionDisplay").setText(
                self.missions[msg.ami_state])

    def shutdown_plugin(self):
        """stop all publisher, subscriber and services
        necessary for clean shutdown"""
        assert (self.node.destroy_client(
            self.set_mission_cli)), "Mission client could not be destroyed"
        assert (self.node.destroy_subscription(
            self.state_sub)), "State subscriber could not be destroyed"
        assert (self.node.destroy_client(
            self.ebs_srv)), "EBS client could not be destroyed"
        assert (self.node.destroy_client(
            self.reset_srv)), "State reset client could not be destroyed"
        # Note: do not destroy the node in shutdown_plugin as this could
        # cause errors for the Robot Steering GUI. Let ROS 2 clean up nodes
    

def main(args=None):

    rclpy.init(args = args)
    ads_dv = path_planner()
    rclpy.spin(ads_dv)

    ads_dv.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()