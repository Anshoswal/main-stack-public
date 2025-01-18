import numpy as np
import math
import yaml
import time
from controller.controller_packages.utilities import *

class Algorithms():
    def __init__(self, CONFIG_PATH : str,t_start, waypoints_available,CarState_available,store_path_taken,current_waypoints,blue_cones,yellow_cones,pos_x,pos_y,car_yaw,v_curr,integral,vel_error,stoppoints_available,stop_signal,too_close_blue,too_close_yellow):
        self.t_start = t_start
        self.t1 = t_start - 1
        
        self.waypoints_available = waypoints_available
        self.CarState_available = CarState_available
        self.store_path_taken = store_path_taken
        self.current_waypoints = current_waypoints
        self.blue_cones = blue_cones
        self.yellow_cones = yellow_cones
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.car_yaw = car_yaw
        self.v_curr = v_curr
        self.integral = integral
        self.vel_error = vel_error
        

        self.stoppoints_available = stoppoints_available
        self.stop_signal = stop_signal
        self.too_close_blue = too_close_blue
        self.too_close_yellow = too_close_yellow
        controls_config_path = CONFIG_PATH / 'controller.yaml'
        with open(controls_config_path) as file:
            controls_config = yaml.load(file, Loader=yaml.FullLoader)
        self.car_length = controls_config['length_of_car']
        self.max_steer_radians = controls_config['max_steer_radians']
        self.v_ref = controls_config['v_ref']
        self.kp = controls_config['kp']
        self.ki = controls_config['ki']
        self.kd = controls_config['kd']
        self.pure_pursuit = controls_config['pure_pursuit']
        self.stanley = controls_config['stanley']
        self.k_static = controls_config['k_static']
        self.t_runtime = controls_config['t_runtime']
        pass

    def throttle_controller(self):
        if (self.waypoints_available and self.CarState_available) == False:
            self.t_start = time.time()
            #self.get_logger().info(f'Waypoints Available:{self.waypoints_available} CarState available:{self.CarState_available}')
            return ('insufficient_info')

        # Run Node for limited time 
        if time.time() < self.t_start + self.t_runtime :
            # print('Enter Control loop')

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # Just for storing path, car ground truth not required #
            pos_x = self.pos_x
            pos_y = self.pos_y
            self.store_path_taken = np.append(self.store_path_taken, [[pos_x,pos_y]], axis = 0)
            #q = self.carState.pose.pose.orientation

            dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
            self.t1 = time.time()
            # closest_waypoint_index=np.argmin((pos_x-self.x)**2+(pos_y-self.y)**2)
            '''
            If Orange cones aren't visible after being seen by car, the car applies brakes to come to stop
            '''
            if((self.stoppoints_available == True or self.stop_signal == True)):
                throttle = 0
                brake = 1
                self.stop_signal = True
                
            else:
                [throttle,brake,self.integral,self.vel_error,diffn ] = vel_controller2(kp=self.kp, ki=self.ki, kd=self.kd,
                                                                        v_curr=self.v_curr, v_ref=self.v_ref,
                                                                        dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)
                print('tthrotle', throttle)
                print('bbrake',brake)
            # print('close_index',closest_waypoint_index)
            # print('no. of midpoints',self.midpoints.shape)
            # print('paired',self.paired_indexes)
           
            # self.get_logger().info(f"Stop Signal : {self.stop_signal}, Car ")

            throttle = float(throttle)
            brake = float(brake)

            return throttle, brake
        
    def control_pure_pursuit(self):
        steer_pp=0
        mean_change , self.k_dynamic = curvature(self.v_ref,self.current_waypoints)
        x_p=0
        y_p =0
        #print(f"mean change :{mean_change}")
        if not self.too_close_blue and not self.too_close_yellow:
            if (len(self.blue_cones.track)<1 and len(self.yellow_cones.track)<2) or (len(self.blue_cones.track)<2 and len(self.yellow_cones.track)<1):
                print("kaha ja rahe, wapis aao", self.current_waypoints)
                
                # print(steer_pp)
                if len(self.blue_cones) > len(self.yellow_cones):
                    steer_pp = self.max_steer_radians
                    print("case3")
                    #print(f'bluecones:{self.old_detected_blue_cones.markers}yellowcones:{self.old_detected_yellow_cones.markers}')
                else:
                    steer_pp = -self.max_steer_radians
                    print("case4")
                    #print(f'bluecones:{self.old_detected_blue_cones.markers}yellowcones:{self.old_detected_yellow_cones.markers}')
                
            else:
                try:
                    print(self.current_waypoints[:,0])
                    [steer_pp, x_p, y_p,steer,theta] = pure_pursuit(x = self.current_waypoints[:,0], y = self.current_waypoints[:,1], 
                                                        vf=self.v_curr, pos_x=0, pos_y=0, 
                                                        veh_head=0, K = self.k_static, L=self.car_length, MAX_STEER = self.max_steer_radians)
                    print(steer_pp)
                    print(f" Car state {self.pos_x,self.pos_y, self.car_yaw}")
                    print('x_p,y_p',x_p,y_p)
                    print('v_curr',self.v_curr)
                    print('theta,steer',theta,steer)
                except:
                    pass

                
        elif self.too_close_blue:
            steer_pp = self.max_steer_radians
        else:
            steer_pp = -self.max_steer_radians
        steer_pp = float(steer_pp)
        return steer_pp, x_p, y_p
    
    def control_stanley(self):
        
        steer_pp, x_p, y_p, waypoint_index = stanley_steering(final_x=self.current_waypoints[:,0],final_y=self.current_waypoints[:,1],v_curr=self.v_curr,pos_x=self.pos_x,pos_y=self.pos_y,car_yaw = self.car_yaw)
        # print('waypoint',x_p,y_p)
        # self.visualize_cones()
        #self.visualize_pp_waypoint(x_pp = x_p,y_pp = y_p)
        # self.visualize_car(x_coord = pos_x, y_coord = pos_y)
        #print('following',x_p,y_p)
        #print('position',pos_x,pos_y)
        #print('steer',steer_pp,'yaw',car_yaw)
        if(time.time()-self.t_start < 0):
            steer_pp = 0.0
            # carControlsmsg.throttle = throttle
            # carControlsmsg.brake = brake
            # carControlsmsg.steering = steer_pp
        
        if(len(self.yellow_cones) < 3):
            steer_pp = -0.3
        elif(len(self.blue_cones) < 3):
            steer_pp = 0.3

        steer_pp = float(steer_pp)
        return steer_pp, x_p,y_p, waypoint_index