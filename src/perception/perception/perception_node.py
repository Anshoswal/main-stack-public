#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node               

# Add the necessary msg type imports here
from std_msgs.msg import String
 
# Algorithm imports here

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
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
