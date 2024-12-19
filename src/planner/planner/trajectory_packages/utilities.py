from scipy.interpolate import interp1d
import numpy as np
import math
from scipy.spatial import Delaunay
from itertools import combinations
from scipy.spatial.distance import cdist


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

 

def midline_delaunay(blue_cones, yellow_cones):
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
  


def triangle_with_colour(new_current_detected_cones,index):
    new_list3 = [[[0,0,0], [0,0,0], [0,0,0]] for _ in range(len(index))]

    for i in range(len(index)):
        new_list3[i][0] = new_current_detected_cones[index[i][0]]
        new_list3[i][1] = new_current_detected_cones[index[i][1]]
        new_list3[i][2] = new_current_detected_cones[index[i][2]]
        return new_list3
    


def midpoints_from_triangle(new_list3):
    x_mid, y_mid = [], []
    line_list=[]
    #storing length of delaunay edges corresponding to midpoints to later use in cost function
    line_length_list = []
    for triangle in new_list3:
        for i in [0,1]:
            for j in range(i+1,3):
                point_1x=triangle[i][0]
                point_2x=triangle[j][0]
                point_1y=triangle[i][1]
                point_2y=triangle[j][1]   
                line_length = math.sqrt((point_1x-point_2x)**2 + (point_1y-point_2y)**2)                
                if triangle[i][2]!=triangle[j][2]:
                    #2.9 and 4
                    line_mid_x = (triangle[i][0]+triangle[j][0])/2
                    line_mid_y = (triangle[i][1]+triangle[j][1])/2
                    line_list.append([triangle[i][0],triangle[i][1],triangle[j][0],triangle[j][1]])
                    line_length_list.append(line_length)
                    x_mid.append(line_mid_x)
                    y_mid.append(line_mid_y)
    

    return x_mid, y_mid, line_list , line_length_list



def filter_points_by_distance(points, car_position, threshold=2):
    car_x, car_y = car_position
    # Compute distances using vectorized operations
    distances = np.sqrt((points[:, 0] - car_x)**2 + (points[:, 1] - car_y)**2)
    # Filter points whose distance is greater than the threshold
    filtered_points = points[distances > threshold]
    return filtered_points


def evaluate_possible_paths(xy_mid, starting_point, n=NUMBER_OF_WAYPOINTS):
    """
    Generates all possible paths using combinations and sorts the midpoints by distance to minimize latency.
    """
    possible_paths = []
    starting_point = np.array(starting_point)

    # Generate all combinations of n midpoints
    midpoint_combinations = list(combinations(xy_mid, n))

    for combination in midpoint_combinations:
        combination = np.array(combination)

        # Compute distances using SciPy's cdist for vectorized distance calculations
        current_point = starting_point
        sorted_combination = [starting_point]

        while combination.shape[0] > 0:
            distances = cdist([current_point], combination, metric="euclidean").flatten()
            closest_index = np.argmin(distances)
            closest_point = combination[closest_index]

            sorted_combination.append(closest_point)
            combination = np.delete(combination, closest_index, axis=0)
            current_point = closest_point

        possible_paths.append(np.array(sorted_combination))
    
    return possible_paths




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
        


def choose_best_path(self, possible_paths, line_length_list, xy_mid, standard_width=TRACK_WIDTH, angle_weight=ANGLE_WEIGHT, edge_weight=EDGE_WEIGHT, k_path_weight=K_PATH_WEIGHT, expected_path_length_avg=EXPECTED_PATH_LENGTH):
    """
    Selects the best path based on cost function optimization.
    """
    min_path_cost = float('inf')
    best_path = None

    # Preprocess edge lengths into a dictionary for O(1) lookup
    edge_length_dict = {tuple(midpoint): length for midpoint, length in zip(xy_mid, line_length_list)}

    for path in possible_paths:
        path_cost = 0

        # Vectorized angle cost calculation
        vectors = np.diff(path, axis=0)
        unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        cosine_angles = np.einsum('ij,ij->i', unit_vectors[:-1], unit_vectors[1:])
        angle_costs = 1 - cosine_angles  # Angle cost formula
        path_cost += angle_weight * np.sum(angle_costs)

        # Vectorized edge cost calculation
        edge_costs = [
            abs(edge_length_dict[tuple(midpoint)] - standard_width)
            for midpoint in path[1:]
        ]
        path_cost += edge_weight * sum(edge_costs)

        # Path length cost
        path_len_cost = abs(self.path_length_avg(path[1:]) - expected_path_length_avg)
        path_cost += k_path_weight * path_len_cost
        # Check for the minimum path cost
        if path_cost < min_path_cost:
            min_path_cost = path_cost
            best_path = path

    return best_path, min_path_cost


def get_best_path(xy_mid_send , posX , posY , line_length_list):
    xy_mid_send = filter_points_by_distance(xy_mid_send ,(posX,posY))
    possible_paths = evaluate_possible_paths(xy_mid_send , [posX , posY] )#possible paths is currently a list of arrays 
    print("length of possible paths",len(possible_paths))
    best_path,min_path_cost = choose_best_path(possible_paths,line_length_list,xy_mid_send)
    return best_path


def path_length_avg(possible_path):
    """
    Vectorized calculation of average path length.
    """
    differences = np.diff(possible_path, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return np.sum(segment_lengths) / len(segment_lengths)



def calculate_angle_cost(midpoint1, midpoint2, midpoint3):
    vector1 = midpoint2 - midpoint1
    vector2 = midpoint3 - midpoint2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return 1 - cosine_angle



def calculate_edge_length_cost(midpoint, line_length_list, xy_mid, standard_width=3.0):
    return abs(line_length_list[xy_mid.index(midpoint)] - standard_width)


def get_boundary(sorted_blue_cones,sorted_yellow_cones):
    blue_boundary=[]
    yellow_boundary=[]
    for j in range(0,len(sorted_blue_cones)-1):
        if check_track(sorted_blue_cones[j-1,0],sorted_blue_cones[j-1,1],sorted_blue_cones[j,0],sorted_blue_cones[j,1],sorted_blue_cones[j+1,0],sorted_blue_cones[j+1,1],j):
            blue_boundary.append([sorted_blue_cones[j,0],sorted_blue_cones[j,1],sorted_blue_cones[j+1,0],sorted_blue_cones[j+1,1]])
        else:
            break

    for j in range(0,len(sorted_yellow_cones)-1):
        if check_track(sorted_yellow_cones[j-1,0],sorted_yellow_cones[j-1,1],sorted_yellow_cones[j,0],sorted_yellow_cones[j,1],sorted_yellow_cones[j+1,0],sorted_yellow_cones[j+1,1],j):
            yellow_boundary.append([sorted_yellow_cones[j,0],sorted_yellow_cones[j,1],sorted_yellow_cones[j+1,0],sorted_yellow_cones[j+1,1]])
        else:
            break
    return blue_boundary,yellow_boundary

def check_track(x0,y0,x1,y1,x2,y2,index):
    """
    #Checks all the line segments of our boundary with angle and length constraints to avoid abnormal boundaries
    """
    
    #Calculate the length of a line given its start and end points.
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    #Calculate the angle between two consecutive lines.
    # Vectors of the two lines
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x1 - x0, y1 - y0])

    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    # Avoid division by zero
    if v1_norm == 0 or v2_norm == 0:
        return 0

    # Compute cosine of the angle using the dot product
    dot_product = np.dot(v1, v2)
    cos_theta = dot_product / (v1_norm * v2_norm)

    # Clip cos_theta to avoid numerical errors outside the range of arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Return the angle in degrees
    angle = np.degrees(np.arccos(cos_theta))
    if index == 0:
        return True
    elif index>0 and length<3 and angle<30:
        return True
    else:
        return False
    

def quaternionToYaw(self,x,y,z,w):
    yaw = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    return yaw
