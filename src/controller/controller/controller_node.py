#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node   
import yaml   
from pathlib import Path 
import time
import math
from math import pi


# Add the necessary msg type imports here
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose, TwistWithCovarianceStamped, Point
from visualization_msgs.msg import Marker,MarkerArray
from eufs_msgs.msg import CanState
from eufs_msgs.srv import SetCanState
from eufs_msgs.msg import ConeArrayWithCovariance
from eufs_msgs.msg import CarState
from dv_msgs.msg import Track
from dv_msgs.msg import PointArray, Track, Cone

# from rclpy.duration import Duration
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
 
# Algorithm imports here
from controller.controller_algorithms import Algorithms
from controller.controller_packages.utilities import *

# Define ROOT 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # /home/your_name/path/to/IITBDV-main-stack/src/controller/controller
CONFIG_PATH = ROOT / 'config'

# Ensure the config path exists (optional check)
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config folder not found at {CONFIG_PATH}")


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.pos_x = 0
        self.pos_y = 0
        self.car_yaw = 0
        self.store_path_taken = np.array([[0,0]])
        self.integral = 0
        self.vel_error = 0
        self.t_start = time.time()
        self.t1 = time.time() - 1
        self.CarState_available = False
        self.waypoints_available = False
        self.stoppoints_available = False
        self.stop_signal = False
        self.current_waypoints = None
        self.too_close_blue = False
        self.too_close_yellow = False

        # parameters here
        with open(CONFIG_PATH / "controller.yaml", "r") as yaml_file:
            self.controller_config_data = yaml.safe_load(yaml_file)
        self.period = self.controller_config_data['period']
        self.t_runtime = self.controller_config_data['t_runtime']
        self.subscribe_carstate = self.controller_config_data['get_carstate']
        self.pure_pursuit = self.controller_config_data['pure_pursuit']
        self.stanley = self.controller_config_data['stanley']
        self.min_normal_dist = self.controller_config_data['min_normal_distance']
        self.max_angle_diff = self.controller_config_data['max_angle_diff']

        # subscribers here 
        with open(CONFIG_PATH / "topics.yaml", "r") as yaml_file:
            self.controller_topic_data = yaml.safe_load(yaml_file)
        #for carstate
        
        #declare params
        self.declare_parameter('data_source', 'simulator_slam')   # Declare the platform being used, default is eufs
        self.declare_parameter('platform', 'eufs')
        # Get the parameter values
        self.platform = self.get_parameter('platform').get_parameter_value().string_value
        self.data_source = self.get_parameter('data_source').get_parameter_value().string_value

        if self.data_source not in ['sim_slam', 'ground_truth','sim_perception','slam','perc_ppc','slam_ppc']: 
            self.get_logger().error("Invalid system for controller parameter. Choose either 'sim_slam', 'ground_truth','sim_perception','slam','perc_ppc','slam_ppc'. Shutting the Node down...")
            self.destroy_node() 


        if self.platform not in ['eufs', 'bot']: 
            self.get_logger().error("Invalid system for controller parameter. Choose either 'eufs' or 'bot'. Shutting the Node down...")
            self.destroy_node() 


        if self.fixed_frame:
            self.car_state_topic = self.controller_topic_data['state']['topic']
            self.car_state_dtype = self.controller_topic_data['state']['data_type']
            self.car_state_subscription = self.create_subscription(
                eval(self.car_state_dtype),
                self.car_state_topic,
                self.get_carstate,
                10
            )
        
        if self.platform == "bot":
            self.velocity_topic = self.controller_topic_data['state']['topic']
            self.velocity_dtype = self.controller_topic_data['state']['data_type']      
            self.car_state_subscription = self.create_subscription(
                self.velocity_dtype,
                self.velocity_topic,
                self.get_rpmdata,
                10
            )


        #for waypoints and stoppoints
        self.planner_waypoints_topic = self.controller_topic_data['waypoints']['topic']
        self.planner_waypoints_dtype = self.controller_topic_data['waypoints']['data_type']
        self.waypoints_subscription = self.create_subscription(
            eval(self.planner_waypoints_dtype),                       
            self.planner_waypoints_topic,                  
            self.store_waypoints,     
            10                            
        )
        self.planner_stopping_topic = self.controller_topic_data['stopping']['topic']
        self.planner_stopping_dtype = self.controller_topic_data['stopping']['data_type']
        self.waypoints_subscription = self.create_subscription(
            eval(self.planner_stopping_dtype),                       
            self.planner_stopping_topic,                  
            self.store_stoppoints,     
            10                            
        )

        #for cones
        self.blue_cones_topic = self.controller_topic_data['cones']['blue']['topic']
        self.yellow_cones_topic = self.controller_topic_data['cones']['yellow']['topic']
        self.cones_dtype = self.controller_topic_data['cones']['blue']['data_type']
        self.cones_subscription = self.create_subscription(
            eval(self.cones_dtype),                       
            self.blue_cones_topic,                  
            self.store_blue_cones,     
            10                            
        )

        self.cones_subscription = self.create_subscription(
            eval(self.cones_dtype),                       
            self.yellow_cones_topic,                  
            self.store_yellow_cones,     
            10                            
        )
        #for boundary
        self.boundary_topic = self.controller_topic_data['boundary']['topic']
        self.boundary_dtype = self.controller_topic_data['boundary']['data_type']
        self.boundary_subscription = self.create_subscription(
            eval(self.boundary_dtype),
            self.boundary_topic,
            self.boundary_constraints,
            10
        )
        #timer callback
        self.timer = self.create_timer(self.period, self.control_callback)

        # publishers here
        self.to_vcu_publisher_topic = self.controller_topic_data[self.platform]['command']['topic']
        self.to_vcu_publisher_dtype = self.controller_topic_data[self.platform]['command']['data_type']
        self.publish_cmd = self.create_publisher(self.to_vcu_publisher_dtype, self.to_vcu_publisher_topic, 5)
        
        

    def store_waypoints(self, msg):
        self.current_waypoints = np.array([[0,0]])
        for point in msg.points:
            x = point.x
            y = point.y
            self.current_waypoints = np.append(self.current_waypoints, [[x,y]], axis=0)
        self.current_waypoints = self.current_waypoints[1:]
        self.waypoints_available = True
        #print('waypoints',self.current_waypoints)

        return None
    
    def store_stoppoints(self,msg):
        if(self.stoppoints_available == False):
            self.stoppoints_available = True
        return None
    
    def store_blue_cones(self, data):
        self.blue_cones=data
        return None
    
    def store_yellow_cones(self,data):
        self.yellow_cones=data
        return None
    
    def boundary_constraints(self,data):#to check for boundary constraints incase the car is too close to the edge
        self.too_close_blue,self.too_close_yellow = check_boundary(data,self.too_close_blue,self.too_close_yellow,self.pos_x,self.pos_y,self.car_yaw, self.min_normal_dist,self.max_angle_diff)
        if self.too_close_blue:
            self.get_logger().info(f'Too close to blue boundary')
        if self.too_close_yellow:
            self.get_logger().info(f'Too close to yellow boundary')
        return None
    
    def get_carstate(self,data):#to get car state of car in case of the frame is fixed
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        x,y,z,w = data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w
        self.car_yaw = quaternionToYaw(x,y,z,w)
        self.carState = data
        self.v_curr=np.sqrt(self.carState.twist.twist.linear.x**2 + self.carState.twist.twist.linear.y**2)
        self.CarState_available = True
        return None 
    
    def get_rpmdata(self, msg):
            self.v_curr = (msg.data*2*pi*self.wheel_rad)/60
            return None

    def control_callback(self):
        if (self.waypoints_available and self.CarState_available) == False:
            self.t_start = time.time()
            self.get_logger().info(f'Waypoints Available:{self.waypoints_available} CarState available:{self.CarState_available}')
            return
        else:
            if time.time() < self.t_start + self.t_runtime :
                control_callback = Algorithms(CONFIG_PATH,self.t_start,self.waypoints_available,self.CarState_available,self.store_path_taken,self.current_waypoints,self.blue_cones,self.yellow_cones,self.pos_x,self.pos_y,self.car_yaw,self.v_curr,self.integral,self.vel_error,self.stoppoints_available,self.stop_signal,self.too_close_blue,self.too_close_yellow)
                self.throttle,  self.brake = control_callback.throttle_controller()
                self.steer_pp, self.x_p, self.y_p = control_callback.control_pure_pursuit()
                self.get_logger().info(f'Speed:{self.v_curr:.4f} Accn:{float(self.throttle - self.brake):.4f} Steer:{float(-self.steer_pp):.4f} Time:{time.time() - self.t_start:.4f}')
                self.send_to_vcu()
            else :
                self.get_logger().info(f'Time Finished')
                raise SystemExit
        return None

    def send_to_vcu(self):
        # Send the information to the topic
        control_msg = AckermannDriveStamped()
        control_msg.drive.steering_angle = float(self.steer_pp)
        control_msg.drive.acceleration = float(self.throttle - self.brake)
        # control_msg.drive.acceleration = 0.05
        self.publish_cmd.publish(control_msg)
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
