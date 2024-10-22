#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node               

# Add the necessary msg type imports here
from std_msgs.msg import String
 
# Algorithm imports here

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
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
