#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node
from pathlib import Path  
import yaml 
import concurrent  
import time     
from math import pi, cos, sin   

# Add the necessary msg type imports here
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from dv_msgs.msg import Track, Cone, ObservationRangeBearing, SingleRangeBearingObservation  
    
    
class VisualizerNode(Node):  
    
    def __init__(self):
            
        super().__init__('visualizer_node')
        self.get_logger().info("Visualizer Node has been started.")
        
        # Subscribers
        self.cones_seen_sub = self.create_subscription(Track, '/perception/cones', self.cones_seen_callback, 10)
        
        # Publishers
        self.cones_seen_pub = self.create_publisher(MarkerArray, '/perception/cones_viz', 10)
        
    def cones_seen_callback(self, msg):
        
        self.ConesSeen = msg
        self.perception_visualization()
           
    
    def perception_visualization(self, ConesSeen):
        
        ConesSeen = Track()
        cones_seen_array = MarkerArray()
        
        for cone in enumerate(ConesSeen.track):
            marker = Marker()
            marker.header.frame_id = "base_footprint"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "my_namespace"
            marker.id = cone
            marker.type = 1
            marker.action = 0
            marker.pose.position.x = (cone.location.x)*cos(cone.location.y)
            marker.pose.position.y = (cone.location.x)*sin(cone.location.y)
            marker.pose.position.z = 0.0    # cone.location.z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            # marker.pose.orientation.w = 1.0
            marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.a = 1.0
            if cone.color == 0:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif cone.color == 1:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif cone.color == 3: # 3 = small orange
                marker.color.r = 0.945
                marker.color.g = 0.353
                marker.color.b = 0.134
            elif cone.color == 2: # 2 = big orange
                marker.color.r = 1.0
                marker.color.g = 0.58
                marker.color.b = 0.44
            elif cone.color == 4:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            
            cones_seen_array.markers.append(marker)

        self.viz_depths.publish(cones_seen_array)