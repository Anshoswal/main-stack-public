#!/usr/bin/env python3
import math
# Import the necessary libraries here
import rclpy                
from rclpy.node import Node   
import yaml   
from pathlib import Path         
from planner.Midline_delaunay import Midline_delaunay
from planner.trajectory_packages.utilities import slam_cones , groundTruth_cones , perc_cones , distance_cones , quaternionToYaw
# Add the necessary msg type imports here
from std_msgs.msg import String
import numpy as np
# Algorithm imports here
from geometry_msgs.msg import Point
from dv_msgs.msg import PointArray
from eufs_msgs.msg import ConeArrayWithCovariance
from eufs_msgs.msg import CarState
# Define ROOT 
# Get path to the config folder
PACKAGE_ROOT = Path(__file__).resolve().parent  # get the path to the package
CONFIG_PATH = PACKAGE_ROOT / 'config'      # path to the config folder


# Ensure the config path exists (optional check)
# if not CONFIG_PATH.exists():
#     raise FileNotFoundError(f"Config folder not found at {CONFIG_PATH}")

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')


        #Initilizing state variables
        self.posX = 0
        self.posY = 0
        self.car_yaw = 0
        self.CarState_available = False
        self.waypoints = []
        #Initilizing cone variables
        self.blue_cones = []
        self.yellow_cones = []
        self.track_available = False
        self.orange_cones = []
        self.big_orange_cones = []
        self.distance_blue = []
        self.distance_yellow  = []
        self.slam_blue_cones = []
        self.slam_yellow_cones = []
        self.slam_big_orange_cones = []
        self.slam_orange_cones = []
        self.waypoints_topic = []
        # perception_config_data here
        with open(CONFIG_PATH / "topics.yaml", "r") as yaml_file:
            self.planner_config_topic = yaml.safe_load(yaml_file)
        with open(CONFIG_PATH / "planner.yaml", "r") as yaml_file:
            self.planner_config = yaml.safe_load(yaml_file)
        self.LENGTH_OF_CAR = self.planner_config['LENGTH_OF_CAR']
        self.FOV = math.radians(self.planner_config['FOV'])
        self.FOV_RADIUS = self.planner_config['FOV_RADIUS']
        self.semi_major_axis = self.planner_config['ellipse_dimensions']['a']
        self.semi_minor_axis = self.planner_config['ellipse_dimensions']['b']
        self.PERCEPTION_DISTANCE = self.planner_config['PERCEPTION_DISTANCE']
        # Declare the parameters 
        self.declare_parameter('data_source', 'simulator_slam')   # Declare the platform being used, default is eufs
        self.declare_parameter('platform', 'eufs')

        # Get the parameter values
        self.platform = self.get_parameter('platform').get_parameter_value().string_value
        self.data_source = self.get_parameter('data_source').get_parameter_value().string_value


        # Raise an error and kill the node if the platform is not bot or eufs
        if self.data_source not in ['sim_slam', 'ground_truth','sim_perception','slam','perc_ppc','slam_ppc']: 
            self.get_logger().error("Invalid system parameter. Choose either 'sim_slam', 'ground_truth','sim_perception','slam','perc_ppc','slam_ppc'. Shutting the Node down...")
            self.destroy_node() 


        if self.platform not in ['eufs', 'bot']: 
            self.get_logger().error("Invalid system parameter. Choose either 'eufs' or 'bot'. Shutting the Node down...")
            self.destroy_node() 


        #loading topics
        self.set_topic_subscriber(self.platform , self.data_source)
        self.set_dataType_subscriber(self.platform , self.data_source)
        self.set_topic_publisher()
        self.set_dataType_publisher()

        # Define the QoS profile
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)


        # subscribers here 
        #note: can do if else here fro slam wince in that both cones and car state are published on the same topic
        print('self.cones data type',self.cones_data_type)
        print('self.cones topic ',self.cones_topic)
        self.cones_subscription = self.create_subscription(
            eval(self.cones_data_type),                       
            self.cones_topic,                  
            self.get_map,     
            qos_profile=qos_policy            
        )
        
        self.carState_subscription = self.create_subscription(
            CarState,                     
            self.state_topic ,                  
            self.get_carState,    
            qos_profile=qos_policy                          
        )

        # publishers here
        self.to_controller_publisher_topic = 'planner_topic'
        self.to_controller_publisher = self.create_publisher(eval(self.waypoint_dataType), self.waypoint_topic, 10)




    def get_map(self, data):
        # Algorithm function calls are made here
        # Dispatch dictionary
        
        dispatch = {
            ("eufs", "sim_slam"): lambda: slam_cones(data,blue_cones,yellow_cones,big_orange_cones,orange_cones ,self.slam_blue_cones ,self.slam_yellow_cones ,self.slam_big_orange_cones ,self.slam_orange_cones,self.posX , self.posY ,self. car_yaw,self.FOV,self.FOV_RADIUS,self.semi_major_axis,self.semi_minor_axis),
            ("eufs", "ground_truth"): lambda: groundTruth_cones(data,blue_cones,yellow_cones,big_orange_cones,orange_cones,self.PERCEPTION_DISTANCE),
            ("eufs", "sim_perception"): lambda: perc_cones(data,blue_cones,yellow_cones,big_orange_cones,orange_cones),
            ("eufs", "slam"): lambda: slam_cones(data,blue_cones,yellow_cones,big_orange_cones,orange_cones ,self.slam_blue_cones ,self.slam_yellow_cones ,self.slam_big_orange_cones ,self.slam_orange_cones  ),
            ("bot", "perc_ppc"): lambda: perc_cones(data,blue_cones,yellow_cones,big_orange_cones,orange_cones),
            ("bot", "slam_ppc"): lambda: slam_cones(data,blue_cones,yellow_cones,big_orange_cones,orange_cones ,self.slam_blue_cones ,self.slam_yellow_cones ,self.slam_big_orange_cones ,self.slam_orange_cones ),
        }#assuming for now that slam cones from simulator,virtual slam cones and bot have the same data type and hence the same function

        blue_cones = []
        yellow_cones = []
        big_orange_cones = []
        orange_cones=[]#need to be reinitialized every time

        get_cones_function = dispatch.get((self.platform, self.data_source))

        if get_cones_function:
            blue_cones, yellow_cones, big_orange_cones , orange_cones = get_cones_function()
        else:
            print("Invalid combination of platform and data source.")
       
        distance_blue = distance_cones(blue_cones,self.car_yaw,self.posX,self.posY,self.LENGTH_OF_CAR)
        distance_yellow = distance_cones(yellow_cones,self.car_yaw,self.posX,self.posY,self.LENGTH_OF_CAR)
        self.blue_cones = np.array(blue_cones)
        self.yellow_cones = np.array(yellow_cones)
        self.big_orange_cones = np.array(big_orange_cones)
        self.orange_cones = np.array(orange_cones)
        self.distance_blue = np.array(distance_blue)       
        self.distance_yellow = np.array(distance_yellow)
        midline_delaunay = Midline_delaunay(CONFIG_PATH, self.blue_cones, self.yellow_cones , self.orange_cones , self.big_orange_cones , self.posX , self.posY , self.car_yaw , self.distance_blue ,self.distance_yellow )
        self.waypoints = midline_delaunay.get_waypoints()#send to publisher

        self.waypoints_msg = PointArray()
        for waypoint in self.waypoints:
            point = Point()
            point.x = waypoint[0]
            point.y = waypoint[1]
            point.z = 0.0
            self.waypoints_msg.points.append(point)
        self.send_to_controller(self.waypoints_msg)
        





    def get_carState(self , data):
        self.posX = data.pose.pose.position.x
        self.posY = data.pose.pose.position.y
        x,y,z,w = (data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)
        self.car_yaw = quaternionToYaw(x,y,z,w)
        self.carState = data
        self.CarState_available = True
        return None  


    def send_to_controller(self,waypoints_msg):
        # Send the information to the topic
        self.to_controller_publisher.publish(waypoints_msg)
        pass
        

    def set_topic_subscriber(self, platform , data_source):
        self.cones_topic = self.planner_config_topic[platform]['cones'][data_source]['topic']
        self.state_topic = self.planner_config_topic[platform]['car_state'][data_source]['topic']

    def set_topic_publisher(self):
        #all planner publishers are related to marker arrays
        self.waypoint_topic = self.planner_config_topic["publisher"]["waypoints"]["topic"]
        
    def set_dataType_subscriber(self , platform , data_source):
        self.cones_data_type = self.planner_config_topic[platform]['cones'][data_source]['data_type']

    def set_dataType_publisher(self):
        self.waypoint_dataType = self.planner_config_topic["publisher"]["waypoints"]["data_type"]
        
    def destroy_node(self):
        self.get_logger().info("Node is shutting down. Calculating pipeline statistics...")
        super().destroy_node()
    
  
def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
