#import dimensions from config file
import yaml
import math
import numpy as np
from trajectory_packages.utilities import midline_delaunay , interpolate , check_track , get_boundary ,filter_points_by_distance , evaluate_possible_paths , choose_best_path ,perp_bisect , distances , get_best_path


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
        ''' 
        not taking slam cones...directly taing the blue cones ...will have a spearate function for extracting cones from slam cones for every run
        self.slam_blue_cones = []
        self.slam_yellow_cones = []
        self.slam_big_orange_cones = []
        self.slam_orange_cones = []'''
        self.posY = posX
        self.posX = posY
        self.car_yaw = car_yaw
        self.yellow_boundary = []
        self.blue_boundary = []
        self.BEST_PATH = ppc_config['BEST_PATH']
        self.INTERPOLATION = ppc_config['INTERPOLATION']

    def get_waypoints(self):
        #not writing the stopping part in get_Waypoints 
        #either write in main node or make a separate python file
        f_tire_x = self.posX + self.LENGTH_OF_CAR/2 * math.cos(self.car_yaw)
        f_tire_y = self.posY + self.LENGTH_OF_CAR/2 * math.sin(self.car_yaw)
        self.yellow_boundary , self.blue_boundary = self.boundary()
        try:
            x_mid, y_mid, line_list , line_length_list = midline_delaunay(self.blue_cones, self.yellow_cones)
            xy_mid = np.column_stack((x_mid,y_mid))
            if self.BEST_PATH:
                best_path = get_best_path(xy_mid)
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
        sorted_blue_cones,sorted_yellow_cones = self.get_sorted_blue_cones()
        return get_boundary(sorted_blue_cones,sorted_yellow_cones)


    def get_sorted_blue_cones(self):
        #this type of function alos in utilitiea?
        sorted_dist_blue = np.argsort(self.distance_blue)
        sorted_dist_yellow = np.argsort(self.distance_yellow)
        
        sorted_blue_cones = self.blue_cones[sorted_dist_blue]
        sorted_yellow_cones = self.yellow_cones[sorted_dist_yellow]
        return sorted_blue_cones , sorted_yellow_cones


    