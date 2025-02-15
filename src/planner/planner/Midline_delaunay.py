#import dimensions from config file
import yaml
import math
import numpy as np
from planner.trajectory_packages.utilities import interpolate , check_track , get_boundary ,filter_points_by_distance , evaluate_possible_paths , choose_best_path ,perp_bisect , distances  , triangle_with_colour , midpoints_from_triangle,get_tyre_coordinates
from scipy.spatial import Delaunay

class Midline_delaunay():
    def __init__(self , CONFIG_PATH:str ,blue_cones,yellow_cones , orange_cones , big_orange_cones ,posX , posY ,car_yaw ,platform, distance_blue = None ,distance_yellow = None):
        ppc_config_path = CONFIG_PATH / 'planner.yaml'
        with open(ppc_config_path) as file:
            ppc_config = yaml.load(file, Loader=yaml.FullLoader)
        self.blue_cones = blue_cones
        self.yellow_cones = yellow_cones    
        self.big_orange_cones = big_orange_cones
        self.orange_cones = orange_cones
        self.ellipse_dim = [ppc_config['ellipse_dimensions']['a'] , ppc_config['ellipse_dimensions']['b']]
        self.platform = platform
        self.LENGTH_OF_CAR = ppc_config[self.platform]['LENGTH_OF_CAR']
        self.TRACK_WIDTH = ppc_config['TRACK_WIDTH']
        self.distance_blue = distance_blue####initialize in the individual files for initiliazing perception data or slam data
        self.distance_yellow = distance_yellow####
        self.posY = posX
        self.posX = posY
        self.car_yaw = car_yaw
        self.yellow_boundary = []
        self.blue_boundary = []
        self.BEST_PATH = ppc_config['BEST_PATH']
        self.INTERPOLATION = ppc_config['INTERPOLATION']
        self.line_length_list = []
        self.NUMBER_OF_WAYPOINTS = ppc_config['NUMBER_OF_WAYPOINTS']
        
    def get_waypoints(self):
        #not writing the stopping part in get_Waypoints 
        #either write in main node or make a separate python file

        f_tire_x,f_tire_y = get_tyre_coordinates(self.posX,self.posY,self.LENGTH_OF_CAR,self.car_yaw)
        self.yellow_boundary , self.blue_boundary = self.boundary(self.blue_cones,self.yellow_cones,self.distance_blue,self.distance_yellow)
    
        try:
            x_mid, y_mid, line_list , self.line_length_list = self.delaunay_waypoints(self.blue_cones,self.yellow_cones)
            print("delaunay waypoints :",x_mid)
            xy_mid = np.column_stack((x_mid,y_mid))
            if self.BEST_PATH:
                best_path = self.get_best_path(self.posX,self.posY)
                print("best path",best_path)
        except:
            x_mid, y_mid, mid_point_cones_array, our_points = perp_bisect(self.blue_cones,self.yellow_cones, self.TRACK_WIDTH)
            xy_mid = np.column_stack((x_mid,y_mid))
        xy_mid = np.unique(xy_mid, axis=0)
        if self.BEST_PATH:
            best_path = self.get_best_path(self.posX,self.posY)
            print("best path",best_path)
            if self.INTERPOLATION and len(np.unique(x_mid))>1:
                distances_from_midpoints = distances(f_tire_x, f_tire_y, x_mid=best_path[:][0], y_mid=best_path[:][1])
                xy_mid_send = interpolate(x_mid=best_path[:][0], y_mid=best_path[:][1], distances=distances_from_midpoints)
        
        #xy_mid_send = xy_mid

        if self.INTERPOLATION and len(np.unique(x_mid))>1:
            distances_from_midpoints = distances(f_tire_x, f_tire_y, x_mid=x_mid, y_mid=y_mid)
            xy_mid = interpolate(x_mid=x_mid, y_mid=y_mid, distances=distances_from_midpoints)
        return xy_mid#alter later according to the switch
        
            



    def boundary(self,blue_cones,yellow_cones,distance_blue,distance_yellow):
        sorted_blue_cones,sorted_yellow_cones = self.get_sorted_cones(blue_cones,yellow_cones,distance_blue,distance_yellow)
        return get_boundary(sorted_blue_cones,sorted_yellow_cones)


    def get_sorted_cones(self,blue_cones,yellow_cones,distance_blue,distance_yellow):
        #this type of function alos in utilitiea?
        sorted_dist_blue = np.argsort(distance_blue)
        sorted_dist_yellow = np.argsort(distance_yellow)
        
        sorted_blue_cones = blue_cones[sorted_dist_blue]
        sorted_yellow_cones = yellow_cones[sorted_dist_yellow]
        return sorted_blue_cones , sorted_yellow_cones
    

    def delaunay_waypoints(self,blue_cones,yellow_cones):
        new_current_detected_cones = np.append(blue_cones,yellow_cones, axis = 0)
        new_current_detected_cones = np.array(new_current_detected_cones)
    
        #Obtaining the delaunay triangles and list of vertices
        triangulation = Delaunay(new_current_detected_cones[:,:2])
        index=triangulation.simplices

        #matching the traingles to the cones' indices to use the colour information
        new_list3 = triangle_with_colour(new_current_detected_cones,index)

        #obtaining midpoint list from triangles
        x_mid, y_mid, line_list , line_length_list = midpoints_from_triangle(new_list3)

        #conversion to numpy array
        x_mid = np.array(x_mid)
        y_mid = np.array(y_mid)
        line_list = np.array(line_list)
        line_length_list = np.array(line_length_list)
        return x_mid, y_mid, line_list , line_length_list
    

    def get_best_path(self,posX,posY):
        xy_mid_send = filter_points_by_distance(xy_mid_send ,(posX,posY))
        possible_paths = evaluate_possible_paths(xy_mid_send , [posX , posY],self.NUMBER_OF_WAYPOINTS )#possible paths is currently a list of arrays 
        print("length of possible paths",len(possible_paths))
        best_path,min_path_cost = choose_best_path(possible_paths ,self.line_length_list ,xy_mid_send)
        return best_path


    