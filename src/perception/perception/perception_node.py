#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node
import yaml        
from pathlib import Path     

# Add the necessary msg type imports here
from std_msgs.msg import String
from utils.msg_utils.to_slam_utils import send_to_SLAM
 
# Algorithm imports here


# Define ROOT 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # /home/your_name/path/to/IITBDV-main-stack/src/perception/perception

# perception_config_data here
with open(ROOT / "perception.yaml", "r") as yaml_file:
    perception_config_data = yaml.safe_load(yaml_file)

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        # parameters here
        
        
        # subscribers here 
        self.cam_subscriber = 'cam_topic'
        self.perception_subscription = self.create_subscription(
            String,                       
            self.subscriber,                  
            self.cam_callback,       
            10                            
        )
        self.perception_subscription
        
        # publishers here
        self.to_slam_publisher_topic = 'perception_topic'
        self.to_slam_publisher = self.create_publisher(String, self.send_to_slam, 10)
        

    def cam_callback(self, msg):
        # Algorithm function calls will be made here
        pass

    def send_to_slam(self):
        # Send the information to the topic
        pass

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
