#import dimensions from config file
import yaml
import math
import numpy as np
from trajectory_packages.utilities import interpolate , check_track , get_boundary ,filter_points_by_distance , evaluate_possible_paths , choose_best_path ,perp_bisect , distances , get_best_path , triangle_with_colour , midpoints_from_triangle
from scipy.spatial import Delaunay

class Midline_delaunay():
    def __init__(self , CONFIG_PATH:str ,blue_cones,yellow_cones , orange_cones , big_orange_cones ,distance_blue,distance_yellow, posX , posY ,car_yaw , ):
        ppc_config_path = CONFIG_PATH / 'planner.yaml'
        with open(ppc_config_path) as file:
            ppc_config = yaml.load(file, Loader=yaml.FullLoader)
        self.blue_cones = blue_cones
        self.yellow_cones = yellow_cones    
        self.big_orange_cones = big_orange_cones
        self.orange_cones = orange_cones
        self.ellipse_dim = [ppc_config['ellipse_dimensions']['a'] , ppc_config['ellipse_dimensions']['b']]
        self.LENGTH_OF_CAR = ppc_config['LENGTH_OF_CAR']
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

        
    def get_waypoints(self):
        #not writing the stopping part in get_Waypoints 
        #either write in main node or make a separate python file
        f_tire_x = self.posX + self.LENGTH_OF_CAR/2 * math.cos(self.car_yaw)
        f_tire_y = self.posY + self.LENGTH_OF_CAR/2 * math.sin(self.car_yaw)
        self.yellow_boundary , self.blue_boundary = self.boundary()
        try:
            x_mid, y_mid, line_list , self.line_length_list = self.midline_delaunay()
            xy_mid = np.column_stack((x_mid,y_mid))
            if self.BEST_PATH:
                best_path = self.get_best_path()
                print("best path",best_path)
            print("xy_mid try",xy_mid)
        except:
            x_mid, y_mid, mid_point_cones_array, our_points = perp_bisect(self.blue_cones,self.yellow_cones, self.TRACK_WIDTH)
            xy_mid = np.column_stack((x_mid,y_mid))

        '''
        This below part of code is used to interpolate points
        '''
        if self.INTERPOLATION and len(np.unique(x_mid))>1:
            distances_from_midpoints = distances(f_tire_x, f_tire_y, x_mid=best_path[:][0], y_mid=best_path[:][1])
            xy_mid_send = interpolate(x_mid=best_path[:][0], y_mid=best_path[:][1], distances=distances_from_midpoints)
        
            



    def boundary(self):
        """
        """
        sorted_blue_cones,sorted_yellow_cones = self.get_sorted_cones()
        return get_boundary(sorted_blue_cones,sorted_yellow_cones)


    def get_sorted_cones(self):
        #this type of function alos in utilitiea?
        sorted_dist_blue = np.argsort(self.distance_blue)
        sorted_dist_yellow = np.argsort(self.distance_yellow)
        
        sorted_blue_cones = self.blue_cones[sorted_dist_blue]
        sorted_yellow_cones = self.yellow_cones[sorted_dist_yellow]
        return sorted_blue_cones , sorted_yellow_cones
    

    def midline_delaunay(self):
        new_current_detected_cones = np.append(self.blue_cones,self.yellow_cones, axis = 0)
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
    

    def get_best_path(self):
        xy_mid_send = filter_points_by_distance(xy_mid_send ,(self.posX,self.posY))
        possible_paths = evaluate_possible_paths(xy_mid_send , [self.posX , self.posY] )#possible paths is currently a list of arrays 
        print("length of possible paths",len(possible_paths))
        best_path,min_path_cost = choose_best_path(possible_paths ,self.line_length_list ,xy_mid_send)
        return best_path


    