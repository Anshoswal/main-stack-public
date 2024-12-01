from scipy.interpolate import interp1d
import numpy as np

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