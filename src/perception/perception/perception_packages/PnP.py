from audioop import mul
import numpy as np
import cv2 as cv

"""
To propagate keypoints via perspective transform

"""

def perspective(l_kpts,projection, bline):
    model_points = (25.4)*np.array([
                                (2.5, 12.0, 0.0),             
                                (1.75, 8.4, 0.0),        
                                (3.25, 8.4, 0.0),     
                                (1.125, 5.4, 0.0),
                                (3.875, 5.4, 0.0),      
                                (0.0, 0.0, 0.0),    
                                (5.0, 0.0, 0.0)      
                            ])
    dist = np.array([0.0,0.0,0.0,0.0,0.0])

    #mtx=np.array([(1000.0,0,1500.0),(0,1000.0,1000.0),(0,0,1)])

    #mtx=np.array([(1940.513916015625,0,980.1828002929688),(0,1940.513916015625,487.65869140625),(0,0,1)])  #old zed params - gives low error
    #mtx=np.array([(530,0,392.5),(0,530,392.5),(0,0,1)])## FSDS Params (old)

    mtx = mtx=np.array([(392,0,392.5),(0,391,392.5),(0,0,1)]) # FSDS Params new




    (success,rotation_vector, translation_vector) = cv.solvePnP(model_points, l_kpts, mtx,dist)
    rodri, jac = cv.Rodrigues(rotation_vector)
    if projection==1:
        translation_vector[0]=translation_vector[0]-bline
        r_kpts,jacobian=cv.projectPoints(model_points,rotation_vector,translation_vector,mtx,dist)
        r_kpts=np.squeeze(r_kpts,axis=1)
        return r_kpts