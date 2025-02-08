from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import  LogInfo
import yaml
from pathlib import Path

# Define ROOT 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # /home/your_name/path/to/IITBDV-main-stack/src/master/launch

# perception_config_data here
with open(ROOT / "master.yaml", "r") as yaml_file:
    master = yaml.safe_load(yaml_file)
    
stack       =    master['stack'][master['stack_i']]
track       =    master['track'][master['track_i']]
perception  =    master['perception'][master['perception_i']]
slam        =    master['slam'][master['slam_i']]
planner     =    master['planner'][master['planner_i']]
controller  =    master['controller'][master['controller_i']]

def generate_launch_description():
    ld = LaunchDescription()
    
    if perception != "none":
        perception_node = Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            parameters=[               
                {'pipeline': perception},
                {'platform': stack}  
            ]
        )
        ld.add_action(perception_node)
    
    if slam != "none":
        slam_node = Node(
            package='slam',
            executable='slam',
            name='slam_node',
            parameters=[               
                {'pipeline': slam},
                {'platform': stack}  
            ],
        )
        ld.add_action(slam_node)
        
    if controller != "none":
        control_node = Node(
            package='controller',
            executable='controller',
            name='control_node',
            parameters=[               
                {'pipeline': controller},
                {'platform': stack}  
            ],
        )
        ld.add_action(control_node)
        
    if planner != "none":
        path_planner_node = Node(
            package='planner',
            executable='planner',
            name='planner_node',
            parameters=[               
                {'pipeline': planner},
                {'platform': stack}  
            ],
        )
        ld.add_action(path_planner_node)
        
    return ld