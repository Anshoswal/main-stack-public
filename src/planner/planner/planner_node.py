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
ROOT = FILE.parents[0] # /home/your_name/path/to/IITBDV-main-stack/src/planner/planner

# perception_config_data here
with open(ROOT / "planner.yaml", "r") as yaml_file:
    planner_config_data = yaml.safe_load(yaml_file)

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        # parameters here
        
        
        # subscribers here 
        self.slam_subscriber_topic = 'slam_topic'
        self.slam_subscription = self.create_subscription(
            String,                       
            self.slam_subscriber_topic,                  
            self.planner_callback,     
            10                            
        )
        self.slam_subscription
        
        # publishers here
        self.to_controller_publisher_topic = 'planner_topic'
        self.to_controller_publisher = self.create_publisher(String, self.send_to_controller, 10)
        

    def planner_callback(self, msg):
        # Algorithm function calls will be made here
        pass

    def send_to_controller(self):
        # Send the information to the topic
        pass

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
