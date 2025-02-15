#!/usr/bin/env python3

# Import the necessary libraries here
import rclpy                
from rclpy.node import Node  
from pathlib import Path  
import yaml 
import concurrent  
import time        
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

# Add the necessary msg type imports here
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from dv_msgs.msg import Track, Cone, ObservationRangeBearing, SingleRangeBearingObservation
from perception.utils.msg_utils.to_slam_utils import vis
from perception.utils.perc_utils import *
from sensor_msgs.msg import PointCloud2
 
# Algorithm imports here
from perception.mono import MonoDepth, MonoPipeline
from perception.fusion import FusionDepth, FusionPipeline
from perception.utils.perc_utils import draw_images, update_pos, process_image
from perception.YOLO import LoadYolo

# Get path to the config folder
PACKAGE_ROOT = Path(__file__).resolve().parent  # get the path to the package
CONFIG_PATH = PACKAGE_ROOT / 'config'      # path to the config folder

# Ensure the config path exists (opftional check)
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config folder not found at {CONFIG_PATH}")

# Perception node with all pipeline functions
class PerceptionNode(Node):
    
    def __init__(self):
        super().__init__('perception_node')

        self.left_image_msg = None
        self.right_image_msg = None
        self.left_image = None
        self.right_image = None
        self.left_boxes = None
        self.right_boxes = None
        self.left_num = 0
        self.right_num = 0

        self.frame_number = 0
        self.pipeline_outputs = {}
        self.time_elapsed = 0
        self.pipeline_times = {}
        self.execution_counts = {}
        
        # # Create mutually exclusive callback groups
        self.callback_group_left = MutuallyExclusiveCallbackGroup()
        self.callback_group_right = MutuallyExclusiveCallbackGroup()
        self.lidar_group = MutuallyExclusiveCallbackGroup()
        # self.callback_group = ReentrantCallbackGroup()
        
        # Declare the parameters 
        self.declare_parameter('platform', 'bot')   # Declare the platform being used, default is eufs
        self.declare_parameter('pipeline','fusion')   # Declare the pipeline being used, default is mono

        # Get the parameter values
        self.platform = self.get_parameter('platform').get_parameter_value().string_value
        pipeline = self.get_parameter('pipeline').value

        # Normalize pipeline to a list
        self.pipeline = [pipeline] if isinstance(pipeline, str) else pipeline

        # Raise an error and kill the node if the platform is not bot or eufs
        if self.platform not in ['bot', 'eufs','carmaker']: 
            self.get_logger().error("Invalid system parameter. Choose either 'bot' or 'eufs'. Shutting the Node down...")
            self.destroy_node() 

        # Load the config file
        with open(CONFIG_PATH / 'perception_config.yaml', 'r') as file:
            self.perc_config = yaml.safe_load(file)
        
        # Assign values to parameters dynamically, Load yolo, set camera topics
        self.set_cam_topics(self.platform)
        self.set_publisher_topic()
        self.image_process_params = self.perc_config['image_process_params']
        self.yolo_left = LoadYolo(PACKAGE_ROOT)
        self.yolo_right = LoadYolo(PACKAGE_ROOT)
        self.temp_vis = vis()
        
        
        # Define available pipelines dynamically
        self.pipeline_map = {
            'mono': self.init_mono_pipeline,
            'dual mono': self.init_dual_mono_pipeline,
            'fusion': self.init_fusion_pipeline,
            'lidar only': self.init_lidar_only_pipeline,
        }
        
        # Validate pipelines and initialize selected ones
        self.active_pipelines = self.initialize_pipelines()

        # Define the QoS profile
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        
        # Subscribers
        self.cam_left_topic = self.create_subscription(
            Image,                       
            self.cam_left_topic,                  
            self.cam_left_callback,       
            qos_profile=qos_policy,
            callback_group=self.callback_group_left                      
        )
        self.cam_right_topic = self.create_subscription(
            Image,
            self.cam_right_topic,
            self.cam_right_callback,
            qos_profile=qos_policy,
            callback_group=self.callback_group_right
        )

        self.lidar_topic = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.get_pcd,
            qos_profile=qos_policy,
            callback_group=self.lidar_group
        )
        
        # Publishers
        self.to_slam_publisher = self.create_publisher(
            Track, 
            self.to_slam_publisher_topic, 
            10
        )
    
    def set_cam_topics(self, platform):
                
        self.cam_left_topic = self.perc_config['cam_topics'][platform]['left']['topic']
        self.cam_right_topic = self.perc_config['cam_topics'][platform]['right']['topic']
        self.lidar_topic = self.perc_config['lidar_topics']['topic']
    
    def set_publisher_topic(self):
        
        self.to_slam_publisher_topic = self.perc_config['perc_publish_topic']['topic']
    
    
    def initialize_pipelines(self):     # Initialize and validate selected pipelines.

        active_pipelines = []
        invalid_pipelines = []

        for pipeline_name in self.pipeline:

            if pipeline_name in self.pipeline_map:
                active_pipelines.append(self.pipeline_map[pipeline_name]())
            else:
                invalid_pipelines.append(pipeline_name)

        if invalid_pipelines:
            self.get_logger().warn(f"Invalid pipeline detected: {invalid_pipelines}")

        self.get_logger().info(f"Active pipelines: {[p.__class__.__name__ for p in active_pipelines]}")
        
        return active_pipelines      
    
    
    def init_mono_pipeline(self):   # Initialize the Mono pipeline and return the instance to be stpred in the active_pipelines list.
    
        self.get_logger().info("Initializing Mono...")
        mono = MonoPipeline(
                    config_path=CONFIG_PATH,
                    platform=self.platform,
                    logger=self.get_logger(),
                    )
        self.get_logger().info("Mono initialization successful")
        return mono

    def init_dual_mono_pipeline(self):  # Initialize the Dual Mono pipeline initialization.
        
        self.get_logger().info("Initializing Dual Mono...")
        # mono
        # Initialize Dual Mono pipeline here
        return None  # Replace with actual pipeline class instance

    def init_fusion_pipeline(self):
        """
        Placeholder for Fusion pipeline initialization.
        """
        self.get_logger().info("Initializing Fusion pipeline...")
        # Initialize Fusion pipeline here
        fusion = FusionPipeline(
                    config_path=CONFIG_PATH,
                    platform=self.platform,
                    logger=self.get_logger(),
                    )
        self.get_logger().info("Mono initialization successful")
        return fusion

    def init_lidar_only_pipeline(self):
        """
        Placeholder for Lidar Only pipeline initialization.
        """
        self.get_logger().info("Initializing Lidar Only pipeline...")
        # Initialize Lidar Only pipeline here
        return None  # Replace with actual pipeline class instance
    
    
    def cam_left_callback(self, msg):
        
        self.left_image_msg = msg  
        self.left_image = process_image(self.left_image_msg)
        self.left_boxes = self.yolo_left.make_bounding_boxes(self.left_image)
        self.left_num += 1

        
        # If both images and their bounding boxes are ready, execute the pipeline
        if self.right_image is not None and self.right_boxes is not None:
            self.sync_callbacks(self.left_image, self.right_image, self.left_boxes, self.right_boxes)     
    
    def cam_right_callback(self, msg):

        self.right_image_msg = msg
        self.right_image = process_image(self.right_image_msg)
        self.right_boxes = self.yolo_right.make_bounding_boxes(self.right_image)
        self.right_num += 1
        
        # If both images and their bounding boxes are ready, execute the pipeline
        if self.left_image is not None and self.left_boxes is not None:
            self.sync_callbacks(self.left_image, self.right_image, self.left_boxes, self.right_boxes)

    def get_pcd(self,msg):
        lidar_height = 0.25
        points = pc2.read_points(msg,skip_nans=True)
        points = np.array([[data[0]+0.59,data[1],data[2]] for data in points if (data[0] > 0) and -lidar_height+0.35 > data[2]> -lidar_height])# X = points[:, :2]  # Use x, y for plane fitting
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
        self.lidar_coords = cones_xy(points_filtered,labels)

    def sync_callbacks(self, left_image, right_image, left_boxes, right_boxes):
        
        self.frame_number += 1
        print(f"Frame number: {self.frame_number}")
        print(f"Left: {self.left_num}, Right: {self.right_num}")
        self.pipeline_executor(left_image, right_image, left_boxes, right_boxes)
        # Reset state after execution
        self.left_image = None
        self.right_image = None
        self.left_boxes = None
        self.right_boxes = None
        
    
    def pipeline_executor(self, left_image, right_image, left_boxes, right_boxes):

        if not self.active_pipelines:
            self.get_logger().warn("No active pipelines to execute.")
            return
        
        if len(self.active_pipelines) == 1:
            # Single pipeline: Execute normally
            self.get_logger().info(f"Executing single pipeline: {self.active_pipelines[0].__class__.__name__}")
            self.execute_single_pipeline(self.active_pipelines[0], left_image, right_image, left_boxes, right_boxes)
            self.send_to_slam()
        else:
            # Multiple pipelines: Execute simultaneously
            self.get_logger().info(f"Executing multiple pipelines: {[p.__class__.__name__ for p in self.active_pipelines]}")
            self.execute_pipelines_simultaneously()
        

    def execute_single_pipeline(self, pipeline, left_image, right_image, left_boxes, right_boxes):
        
        pipeline_name = pipeline.__class__.__name__.lower()
        
        if left_image is not None and right_image is not None:
            self.get_logger().info(f"Processing with {pipeline_name}...")
            try:
                
                # Record start time
                start_time = time.time()

                # Execute the specific pipeline
                if (pipeline_name == 'monopipeline'):
                    depths, thetas, ranges, colors = pipeline.monopipeline(left_image, right_image, self.frame_number, left_boxes, right_boxes, self.platform)
                elif (pipeline_name == 'fusionpipeline'):
                    depths, fusion_d, thetas, ranges, colors = pipeline.fusionpipeline(left_image, right_image, self.frame_number, left_boxes, right_boxes, self.lidar_coords, self.platform)
                elif (pipeline_name == 'dualmonopipeline'):
                    depths, thetas, ranges, colors = pipeline.dualmonopipeline(left_image, right_image, self.frame_number, left_boxes, right_boxes)
                
                elapsed_time = time.time() - start_time

                # print(f"Time taken to execute {pipeline_name}: {elapsed_time:.4f} seconds")
                
                if pipeline_name not in self.pipeline_times:
                    self.pipeline_times[pipeline_name] = 0
                    self.execution_counts[pipeline_name] = 0
                
                self.pipeline_times[pipeline_name] += elapsed_time
                self.execution_counts[pipeline_name] += 1

                # Dynamically store the results for this pipeline
                self.pipeline_outputs[pipeline_name] = {
                    "depths": depths,
                    "depths_using_fusion": fusion_d,
                    "thetas": thetas,
                    "ranges": ranges,
                    "colors": colors
               }
            
            except Exception as e:
                self.get_logger().error(f"Error in {pipeline_name}: {e}")
    

    def send_to_slam(self):
        
        if len(self.active_pipelines) == 1:
            # Single pipeline: Send data to SLAM
            self.get_logger().info(f"Sending data to SLAM from {self.active_pipelines[0].__class__.__name__}")
            try: 
                data = self.pipeline_outputs[self.active_pipelines[0].__class__.__name__.lower()]
            except KeyError:
                self.get_logger().error("No cones detected :(")
                return
            
            thetas = data['thetas']
            ranges = data['ranges']
            colors = data['colors']

            self.temp_vis.send_to_SLAM(thetas, ranges, colors)
        # else:
        #     # Multiple pipelines: Combine data and send to SLAM
        #     self.get_logger().info(f"Sending data to SLAM from multiple pipelines")
        #     combined_data = self.combine_data() # Implement this function, after multi threading is implemented
        #     send_to_SLAM(combined_data)
    

    def execute_pipelines_simultaneously(self):
        """
        Execute multiple pipelines simultaneously using ThreadPoolExecutor.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_pipeline = {
                executor.submit(self.execute_pipeline, pipeline): pipeline
                for pipeline in self.active_pipelines
            }

            for future in concurrent.futures.as_completed(future_to_pipeline):
                pipeline = future_to_pipeline[future]
                try:
                    future.result()  # This will raise an exception if one occurred in the thread
                    self.get_logger().info(f"{pipeline.__class__.__name__} completed successfully.")
                except Exception as e:
                    self.get_logger().error(f"Error in {pipeline.__class__.__name__}: {e}")

        """
        TO DO: Implement the logic to store data from various pipelines as in the single executor, 
        then sync it, and write a function to combine that data or like how to send it to slam
        """

    def destroy_node(self):
        self.get_logger().info("Node is shutting down. Calculating pipeline statistics...")
        
        for pipeline_name, total_time in self.pipeline_times.items():
            count = self.execution_counts.get(pipeline_name, 0)
            average_time = total_time / count if count > 0 else 0
            print(f"Pipeline: {pipeline_name}, Average Time: {average_time:.4f} seconds over {count} executions")
            
        super().destroy_node()



def main(args=None):
    rclpy.init()
    node = PerceptionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node...")
        # Clean up resources and destroy the node
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()