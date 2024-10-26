#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node   
import yaml   
from pathlib import Path         

# Add the necessary msg type imports here
from std_msgs.msg import String
 
# Algorithm imports here

# Define ROOT 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # /home/your_name/path/to/IITBDV-main-stack/src/controller/controller

# perception_config_data here
with open("controller.yaml", "r") as yaml_file:
    controller_config_data = yaml.safe_load(yaml_file)
class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        # parameters here
        
        
        # subscribers here 
        self.planner_subscriber_topic = 'planner_topic'
        self.planner_subscription = self.create_subscription(
            String,                       
            self.planner_subscriber_topic,                  
            self.controller_callback,     
            10                            
        )
        self.planner_subscription
        
        # publishers here
        self.to_vcu_publisher_topic = 'controller_topic'
        self.to_vcu_publisher = self.create_publisher(String, self.send_to_vcu, 10)
        

    def controller_callback(self, msg):
        # Algorithm function calls will be made here
        pass

    def send_to_vcu(self):
        # Send the information to the topic
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
