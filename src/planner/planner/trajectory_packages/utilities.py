from scipy.interpolate import interp1d
import numpy as np
import math

def unique(a:np.ndarray) -> np.ndarray: 
    # Removes repeated midpoints if any
    indexes = np.unique(a, axis=0, return_index=True)[1]
    return [a[index] for index in sorted(indexes)]

def distances(tire_x:float, tire_y:float, x_mid:list, y_mid:list) -> list:
    # Gives the distances of midpoints from the front or back tire
    distances = [np.sqrt((x_mid[i] - tire_x)**2 + (y_mid[i] - tire_y)**2) for i in range(len(x_mid))]
    return distances

def interpolate(x_mid:list, y_mid:list, distances:list) -> np.ndarray:
    # Gives the interpolated path given the midpoints
    sort = np.argsort(distances)
    x_temp=list(x_mid)
    y_temp=list(y_mid)

    for i in range(len(sort)):
        x_mid[i] = x_temp[sort[i]]
        y_mid[i] = y_temp[sort[i]]

    x_mid=np.array(x_mid)
    y_mid=np.array(y_mid)
    xy_mid=np.column_stack((x_mid, y_mid))
    xy_mid = np.array(unique(xy_mid))       #Make a combined list and then find unique as two cones can have same y value  
    x_mid = list(xy_mid[:,0])
    y_mid = list(xy_mid[:,1])
    # When no. of waypoints is 2 or 3, interpolation doesnt work so we add points between the given points by ratios
    if len(x_mid) == 3 and len(y_mid) == 3:
        x_mid.insert(1,(x_mid[0]+x_mid[1])/2)
        y_mid.insert(1,(y_mid[0]+y_mid[1])/2)
        x_mid.insert(3,(x_mid[2]+x_mid[3])/2)     
        y_mid.insert(3,(y_mid[2]+y_mid[3])/2)
    elif len(x_mid) == 2 and len(y_mid) == 2:
        x_mid.insert(1,(2*x_mid[0]+x_mid[1])/3)
        y_mid.insert(1,(2*y_mid[0]+y_mid[1])/3)
        x_mid.insert(2,(x_mid[0]+2*x_mid[2])/3)
        y_mid.insert(2,(y_mid[0]+2*y_mid[2])/3)

    # Create a parameter t for the interpolation
    i = np.arange(len(x_mid))

    # Perform Cubic spline interpolation for x and y separately
    spline_x = interp1d(i, x_mid, kind='cubic')
    spline_y = interp1d(i, y_mid, kind='cubic')

    # Create a finer parameter t for the smooth curve
    interp_i = np.linspace(0, i.max(), 10 * i.max())

    # Calculate the corresponding x and y values for the smooth curve
    x_smooth = spline_x(interp_i)
    y_smooth = spline_y(interp_i)
    x_mid = x_smooth
    y_mid = y_smooth
    xy_mid=np.column_stack((x_mid, y_mid))
    return xy_mid

def midline_delaunay(self, blue_cones, yellow_cones):
        new_current_detected_cones = np.append(blue_cones,yellow_cones, axis = 0)
        new_current_detected_cones = np.array(new_current_detected_cones)
        #print(bluecones_withcolor)
        #print(new_current_detected_cones)
        print("heliiii")
        x_mid, y_mid = [], []
        triangulation = Delaunay(new_current_detected_cones[:,:2])
        index=triangulation.simplices
        new_list3 = [[[0,0,0], [0,0,0], [0,0,0]] for _ in range(len(index))]

        for i in range(len(index)):
            new_list3[i][0] = new_current_detected_cones[index[i][0]]
            new_list3[i][1] = new_current_detected_cones[index[i][1]]
            new_list3[i][2] = new_current_detected_cones[index[i][2]]

        print("hehe")
        #vertices= triangulation.points[c]
        print("SOB")
        # p=0
        q = 0
        # cntr=0
    
        print("NOOOOOOOOOOOo")
        ''' for p in range(len(vertices)):
                found_pair = False
                triangle = vertices[p]
                # if p==len(vertices):
                #     break
                for n in range(0,3):
                    # if found_pair:
                    #     break
                    for i in range(len(new_current_detected_cones)):
                        if triangle[n][0]==new_current_detected_cones[i][0] and triangle[n][1]==new_current_detected_cones[i][1]:
                            #print(p, "yess")
                            new_list3[p][n][0]=new_current_detected_cones[i][0] 
                            new_list3[p][n][1]=new_current_detected_cones[i][1]
                            new_list3[p][n][2]=new_current_detected_cones[i][2]
                            found_pair = True
                            #print(new_list[i])
                            break'''

        line_list=[]
        #print("new list", new_list3)
        
        print("Yooo")
            
        for triangle in new_list3:
            for i in [0,1]:
                for j in range(i+1,3):
                    point_1x=triangle[i][0]
                    point_2x=triangle[j][0]
                    point_1y=triangle[i][1]
                    point_2y=triangle[j][1]   
                    line_length = math.sqrt((point_1x-point_2x)**2 + (point_1y-point_2y)**2)                
                    if 2.0<line_length<14 and triangle[i][2]!=triangle[j][2]:
                        #2.9 and 4
                        line_mid_x = (triangle[i][0]+triangle[j][0])/2
                        line_mid_y = (triangle[i][1]+triangle[j][1])/2
                        line_list.append([triangle[i][0],triangle[i][1],triangle[j][0],triangle[j][1]])
                        x_mid.append(line_mid_x)
                        y_mid.append(line_mid_y)
        

        return x_mid, y_mid, line_list
    
def perp_bisect(self, blue_cones, yellow_cones, TRACK_WIDTH=1.5):
        x_mid, y_mid = [], []
        #trying to get pseudo way points by the perp bisector method
        mid_point_cones_array = np.array([[0,0]])
        if len(blue_cones)>=2:
            perp_cones = blue_cones
            print('case1')
            print(f'bluecones:{blue_cones}yellowcones:{yellow_cones}')
        elif len(yellow_cones)>=2:
            perp_cones = yellow_cones
            print('case2')
            print(f'bluecones:{blue_cones}yellowcones:{yellow_cones}')
        
        dist_cones = []
        for cone in perp_cones:
            dist_cone = math.sqrt((cone[0]**2)+(cone[1]**2))#distance of cones wrt car in car's frame
            dist_cones.append(dist_cone)
        dist_cones = np.array(dist_cones)
        sorted_indices = np.argsort(dist_cones)#sorting wrt the distance
        sorted_perp_cones = [perp_cones[sorted_i] for sorted_i in sorted_indices]#gives us the sorted cones array which will be used for perp bisector so we can take their pairs
        our_points = []
        for i in range(len(sorted_perp_cones)-1):
            mid_point_cones = np.array([(sorted_perp_cones[i][0]+sorted_perp_cones[i+1][0])/2,(sorted_perp_cones[i][1]+sorted_perp_cones[i+1][1])/2])
            mid_point_cones_array = np.append(mid_point_cones_array, [mid_point_cones], axis=0)
            #gives us array of mid points of pair of cones that we take
            #unit direction vector

            magnitude = math.sqrt((sorted_perp_cones[i][0]-sorted_perp_cones[i+1][0])**2 + (sorted_perp_cones[i][1]-sorted_perp_cones[i+1][1])**2)

            unit_perpendicular_vector = np.array([(sorted_perp_cones[i][1]-sorted_perp_cones[i+1][1])/magnitude,(sorted_perp_cones[i+1][0]-sorted_perp_cones[i][0])/magnitude])

            displacement_vector = np.array([unit_perpendicular_vector[0]*TRACK_WIDTH,unit_perpendicular_vector[1]*TRACK_WIDTH])
            #1.5 is approximate half width of road

            possible_waypoints = np.array([[mid_point_cones[0]+displacement_vector[0],mid_point_cones[1]+displacement_vector[1]],[mid_point_cones[0]-displacement_vector[0],mid_point_cones[1]-displacement_vector[1]]])
            dist_from_car = np.array([math.sqrt(possible_waypoints[0][0]**2+possible_waypoints[0][1]**2),math.sqrt(possible_waypoints[1][0]**2+possible_waypoints[1][1]**2)])
            our_point_index = np.argmin(dist_from_car)
            #basically after taking perpendicular bisector we will get two points on that line with the same dist from mid point of cones and we select the point closest to car as it will lie inside the road

            our_point = possible_waypoints[our_point_index]
            our_points.append(our_point)

            print(f'perp_working, waypoint{our_point}')

            x_mid.append(our_point[0])
            y_mid.append(our_point[1])
        mid_point_cones_array = mid_point_cones_array[1:]
        

        return x_mid, y_mid, mid_point_cones_array, our_points
        