import math
import numpy as np
import yaml
from visualization_msgs.msg import Marker,MarkerArray




def quaternionToYaw(x,y,z,w):
        yaw = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
        return yaw


def curvature(waypoints:np.ndarray, k_static, v_ref:float):#calculates average curvature and creates an inverse relation between curvature and lookahead distance
        
        x = waypoints[:,0]
        y = waypoints[:,1]
        angle = []
        change = []
        for i in range(min(len(waypoints)-2,5)):
            angle1 = math.atan2(x[i+1]-x[i], y[i+1]-y[i])
            angle2 = math.atan2(x[i+2]-x[i+1], y[i+2]-y[i+1])
            change.append(abs(angle1 - angle2))
        mean_change = np.mean(change)
        #mean_change = min(0.05,mean_change)

        k_dynamic = k_static - mean_change*7.5
        v_ref_dynamic = v_ref - mean_change*0.5
        v_ref_dynamic = max(v_ref_dynamic,0.5)
        print('curvature factor',mean_change)
        print('target velocity',v_ref_dynamic)
        return mean_change, k_dynamic, v_ref_dynamic

def line_proximity(x1,y1,x2,y2,pos_x,pos_y,yaw):#This function checks the proximity of a car from the boundary line segments in terms of perpendicular distance and angle
       # Line segment vector
        line_vector = np.array([x2 - x1, y2 - y1])
        car_vector = np.array([pos_x - x1, pos_y - y1])

        # Project car vector onto the infinite line to find perpendicular point
        t = np.dot(car_vector, line_vector) / np.dot(line_vector, line_vector)

        # Find the perpendicular point on the extended line
        nearest_x = x1 + t * line_vector[0]
        nearest_y = y1 + t * line_vector[1]

        # Calculate the true perpendicular distance
        normal_distance = np.linalg.norm([pos_x - nearest_x, pos_y - nearest_y])

        # Angle between car's yaw and the line segment's orientation
        #assumes +ve x axis towards the line segment
        line_angle = np.arctan2(line_vector[1], line_vector[0])
        car_angle = np.radians(yaw)
        angle_diff = np.degrees(np.arctan2(np.sin(car_angle - line_angle), np.cos(car_angle - line_angle)))

        return normal_distance, angle_diff



def check_boundary(data,too_close_blue, too_close_yellow, pos_x,pos_y,car_yaw, min_normal_distance, max_angle_diff): #checks line proximity for the boundary
        too_close_blue = False
        too_close_yellow = False
        #print(data)
        for marker in data.markers:
            x1 = marker.points[0].x
            x2 = marker.points[1].x
            y1 = marker.points[0].y
            y2 = marker.points[1].y

            normal_distance, angle_diff = line_proximity(x1,y1,x2,y2,pos_x,pos_y,car_yaw)
            if marker.color.b == 1.0:
                if normal_distance < min_normal_distance and angle_diff > max_angle_diff: #have to import these variables from yaml file
                    too_close_blue = True

            else:
                if normal_distance < min_normal_distance and angle_diff < -max_angle_diff:
                    too_close_yellow = True
        return too_close_blue, too_close_yellow


def vel_controller2(prev_vel_error, v_curr, v_ref, dt, prev_integral, kp, ki, kd):
    error = v_ref - v_curr
    integral = prev_integral+error*dt
    diffn = (error - prev_vel_error)/dt
    pedal = kp * error + ki * integral + kd * diffn

    if pedal > 0:
        throttle = min(pedal,1)
        brake = 0
    else:
        throttle = 0
        brake = min(-pedal,1)
    return [throttle, brake,integral,error,diffn]


def pure_pursuit(x, y, vf, pos_x, pos_y, veh_head ,K, L , MAX_STEER):
    '''
    L - Length of the car (in bicycle model?)
    look-ahead distance => tune minimum_look_ahead, K
    '''
    minimum_look_ahead = 1.5
    look_ahead_dist = minimum_look_ahead + K*vf

    # Remove points which are not of interest i.e the points which have been passed 
    # In first lap this part is redundant because we will only have points which lie ahead of us

    #   Necessary to initialise like this to be able to append using numpy, this point will always get discarded because its distance will be very high
    points_ahead = np.array([[0,0,10000]])  
    for i in range(len(x)):
        heading_vector = [math.cos(veh_head), math.sin(veh_head)]
        look_ahead_vector = [x[i] - pos_x ,y[i] - pos_y ]
        dot_product = np.dot(heading_vector, look_ahead_vector)
        # print(f'Dot:{dot_product}')
        if dot_product < 0:
            continue
        else:
            # Add how close is the distnace of the waypoint to the look ahead distance needed
            dist_waypoint = math.sqrt((x[i] - pos_x)**2 + (y[i] - pos_y)**2)
            points_ahead = np.append(points_ahead, [[x[i],y[i],abs(dist_waypoint - look_ahead_dist)]], axis = 0)
    
    # Remove the extra point added while creating the varible
    points_ahead = points_ahead[1:]
    
    #Select the waypoint which is closest after remo
    final_waypoint_index = np.argmin(points_ahead[:,2])
    waypoint_x = points_ahead[final_waypoint_index,0]
    waypoint_y = points_ahead[final_waypoint_index,1]
    
    # Angle of the vector connecting car to the waypoint
    theta = math.atan((waypoint_y - pos_y)/(waypoint_x - pos_x))

    if (waypoint_y - pos_y)*(waypoint_x - pos_x) < 0:
        if (waypoint_y - pos_y) > 0:
            theta = theta + math.pi
    else:
        if (waypoint_y - pos_y) < 0:
            theta = theta - math.pi

    #alpha_pp - the change in angle that should be made in car heading
    alpha_pp = theta - veh_head 
    waypoint_distance = ((pos_x - waypoint_x)**2 + (pos_y - waypoint_y)**2)**0.5

    steer = math.atan(2*L*math.sin(alpha_pp/waypoint_distance))# steer in radians

    max_steer_radians = MAX_STEER 

    #Clip the steering to max values
    final_steer = max( - max_steer_radians, min(steer , max_steer_radians))
    if abs(final_steer) == MAX_STEER:
         print("max steer from pure persuitttttttttttttttttttttttttttt")
    return [final_steer, waypoint_x, waypoint_y,steer,theta]

def stanley_steering(final_x, final_y, v_curr, pos_x, pos_y, car_yaw):
    """
    Calculate the steering angle based on the Stanley method for path tracking.

    Parameters:
    - final_x, final_y: Arrays of x and y coordinates of the path waypoints.
    - v_curr: Current velocity of the vehicle.
    - pos_x, pos_y: Current x and y position of the vehicle.
    - car_yaw: Current yaw angle of the vehicle in radians.

    Returns:
    - steering_angle: The recommended steering angle in radians.
    - nearest_x: The x coordinate of the nearest path point.
    - nearest_y: The y coordinate of the nearest path point.
    """
    if len(final_x) != len(final_y) or len(final_x) == 0:
        raise ValueError("Path coordinate arrays must be non-empty and of equal length.")
    k_soft = 3 #To prevent steering angle blowing up at low speeds
    k = 0.03 * v_curr# Controller gain
    
    max_steering_angle = np.deg2rad(22)  # Maximum steering angle in radians
    front_axle_offset = 0.75  # Distance from vehicle position to front axle

    # Compute the position of the front tire
    f_tire_x = pos_x + front_axle_offset * math.cos(car_yaw)
    f_tire_y = pos_y + front_axle_offset * math.sin(car_yaw)

    # Compute path slopes and store them
    theta_p = [math.atan2(final_y[i+1] - final_y[i], final_x[i+1] - final_x[i]) for i in range(len(final_x)-1)]
    theta_p.append(math.atan2(final_y[0] - final_y[-1], final_x[0] - final_x[-1]))      

    # Compute the nearest path point to the front tire
    distances = [np.hypot(final_x[i] - f_tire_x, final_y[i] - f_tire_y) for i in range(len(final_x))]
    #theta = [math.atan2(final_y[i]-f_tire_y, final_x[i]-f_tire_x) for i in range(len(final_x))]
    #nearest2 = np.argsort(distances)[:2]
    #min_cte_index = nearest2[0] 
    #x1,y1 = final_x[min_cte_index],final_y[min_cte_index]
    #min_cte_index1 = nearest2[1]
    #x2,y2 = final_x[min_cte_index1],final_y[min_cte_index1]
    #m = (y2-y1)/(x2-x1)
    #c = y2 - x2*m
    #perpd = abs(m * f_tire_x + c - f_tire_y)/math.sqrt(m**2 +1)
    #min_cte = perpd
    #min_cte = (perpd + distances[min_cte_index])/2
    min_cte_index = np.argmin(distances)
    min_cte = distances[min_cte_index]#* math.cos(theta[min_cte_index])  
    # Determine if the front axle should correct to the left or right  
    front_axle_vector = np.array([math.cos(car_yaw), math.sin(car_yaw)])

    nearest_path_vector = np.array([final_x[min_cte_index] - f_tire_x, final_y[min_cte_index] - f_tire_y])
    cross_prod = np.cross(front_axle_vector, nearest_path_vector)
    cte = -min_cte if cross_prod < 0 else min_cte
    #print(f"{min_cte_index1} and {min_cte_index}")
    # if (abs(cte) > 1):
    #     cte=0
    # Calculate steering angle
    yaw_error = theta_p[min_cte_index] - car_yaw
    steering_angle = (yaw_error/5 + math.atan((k * cte) / (v_curr + k_soft)))
    #steering_angle = math.atan((k * cte) / (v_curr + k_soft))
    #print(f"yaw term {yaw_error}, cte: {cte}, cte term{math.atan((k * cte) / (v_curr+1))}, steer :{steering_angle} , index : {min_cte_index}")
    steering_angle = max(-max_steering_angle, min(max_steering_angle, steering_angle))

    return steering_angle, final_x[min_cte_index], final_y[min_cte_index],min_cte_index

