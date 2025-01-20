
from collections.abc import Sequence

import cv2
from cv2 import exp
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import pickle
from skimage.util import view_as_windows
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from perception.perception_packages.metrics import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from perception.perception_packages.triangulator import DepthFinder
from scipy.stats import linregress
from perception.perception_packages.triangulator import *
from perception.perception_packages.projection import *
from perception.Keypoint_Detection.Keypoint import Keypoints
from perception.perception_packages.sift import Features
from perception.perception_packages.PnP import*

def camera_params(camera_type):
    """
    outputs: 
    focal_length: in pixels
    """
    if camera_type == 0:
        focal_length = 392 
        image_width = 785
        image_height = 785

    if camera_type == 1:
        pass

    


def sift_bypass(left_boxes,left_image,kpr_model,right_image):
    triangulator = DepthFinder()
    Sift= Features()

    l_kpts = get_kp_from_bb(left_boxes, left_image, kpr_model,image='left')
    right_kpts=[]
    for i in range(len(l_kpts)):
        right_kpts.append(np.array(perspective(np.array(l_kpts[i],dtype="double"),projection=1, bline=120),dtype='int32'))
    right_bboxes=get_bbox_from_kpts(right_kpts,left_boxes,right_image)
    left_bboxes=np.array(left_boxes,dtype='object')[:,1]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    features = Sift.feature_detect(left_bboxes, right_bboxes, left_image, right_image,draw=0)
    l_match_pts,r_match_pts,left_boxes_new,right_boxes_new,l_kpts_rektnet_new = Sift.feature_matching(features,left_bboxes,right_bboxes,l_kpts,draw=0) 
    depths=[]
    for i in range(len(l_match_pts)):
        depth= triangulator.find_depth(torch.tensor(l_match_pts[i]), torch.tensor(r_match_pts[i]))
        depths.append(depth)

    
    return depths[0]
    
    

def cone_centre(left_boxes,l_kpts):
    cone_centres = []
    for i in range(len(left_boxes)):    
        conec_x = 0
        conec_y = 0
        for j in range(7): 
            conec_x = conec_x + int(l_kpts[i][j][0])
            conec_y = conec_y + int(l_kpts[i][j][1])
        conec_x = conec_x//7 
        conec_y = conec_y//7
        cone_centres.append([conec_x,conec_y])
    return cone_centres

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
    
def bearing(depth,cone_centre, img_shape):
    # focal_length=1000 #for zed (our zed ie zed2i) (apparently, since rostopic echo camera info on the rosbag says 338 for 640x360 images)
    
    # focal_length = 392 # for fsds
    # focal_length = 640 # for carmaker
    focal_length = 448.13386274345095 # for eufs
    # focal_length = 880.8817073599246 # for eufs 1280x720

    centre=(img_shape[0]//2,img_shape[1]//2)
    
    distance=(centre[1]-cone_centre[0])
    
    theta=180*np.arctan(distance/focal_length)/np.pi
    range_2d=float(depth)/(np.cos(theta*(np.pi)/180))  #2D Range
    cone_height = 0.325
    camera_height = 0.8 
    height_diff=camera_height-cone_height
    range_3d=np.sqrt((range_2d**2)+(height_diff**2))
    return theta, range_2d

def get_depth_from_range(range_3d,theta):
    cone_height = 0.325
    camera_height = 0.8 
    height_diff=camera_height-cone_height
    range = np.sqrt((range_3d**2)-(height_diff**2))
    depth = range * (np.cos(theta*(np.pi)/180))
    

    return depth

def get_depth_using_bb_dimensions(left_boxes,left_image,kpr_model,right_image):
    

    depths = []
    for i, (cls,xywh,conf) in enumerate(left_boxes):

        #iterates thru all the cones in an image
        
        #cls,xywh,conf=left_boxes[i]
        w=xywh[2]
        h=xywh[3]
        
        bby=xywh[1]
        

        lx= left_image.shape[1]
        ly=left_image.shape[0]
        
        
        #estimated_depth = 0.498 * (h**(-0.954))   #after 19th march ZED trials results (curve fit)
        #estimated_depth = 0.423/h  #theoretical formula using 1000/720*30.48h

        #FOR USING ZED
        """
        depth_using_w = 0.265 * (w ** (-0.836))
        depth_using_h = 0.498 * (h ** (-0.954)) 
        """

        #FOR USING FSDS
        
        depth_using_w = 0.265 * (w ** (-0.836))
        depth_using_h = 0.338/h



        w_weight = 0.0
        h_weight = 1

        bad_cone = int((bby + h / 2) * ly) > ly or (w * lx) > (h * ly)
        
        if bad_cone:
            
            left_boxes_new = []
            left_boxes_new.append(left_boxes[i])
            estimated_depth = sift_bypass(left_boxes_new,left_image,kpr_model,right_image)
        else:

            estimated_depth = h_weight * depth_using_h + w_weight * depth_using_w
    
        depths.append(estimated_depth)
    
    return depths



def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def imshow(img, caption='image', wait=0):

    cv2.imshow(caption, img)
    cv2.waitKey(wait)

def get_kp_from_bb(left_boxes, left_image, kpr_model,image):
    left_kpts = []
    height, width, _ = left_image.shape
    for idx,conebb in enumerate(left_boxes):

        ''' Get bounding box properties '''
        if image=='left':
            cls, xywh, conf = conebb
            x, y, w, h = xywh
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

        elif image=='right':
            xywh=conebb
            x,y,w,h=xywh
            x1 = int((x - w / 2))
            y1 = int((y - h / 2))
            x2 = int((x + w / 2))
            y2 = int((y + h / 2))

        ''' Extract bounding box and get left(predicted) keypoints '''
        cone_img = left_image[y1:y2,x1:x2]
        cheight, cwidth, _ = cone_img.shape
        kpts = kpr_model.get_keypoints(cone_img)
        kpts = np.array(kpts * [[cwidth,cheight]])

        ''' Get left(predicted) image keypoints'''
        left_pts = []
        for pt in kpts:
            if image=='left':
                cvpt = (int(pt[0]+((x - w / 2) * width)), int(pt[1]+((y - h / 2) * height)))
                left_pts.append(list(cvpt))
            elif image=='right':
                cvpt = (int(pt[0]+((x - w / 2))), int(pt[1]+((y - h / 2))))
                left_pts.append(list(cvpt))      
        left_kpts.append(left_pts)
    
    return left_kpts

def draw_propagate(left_kpts, right_kpts, left_img, right_img, line=False, annots=None):
    
    img_pair = np.concatenate((left_img, right_img), axis=1)    
    width = left_img.shape[1]

    colors = [[255, 255, 255], [147, 20, 255], [255, 0, 0], [0, 0, 0], [0, 100, 0], [211,0,148], [0, 0, 255], [255, 255, 255], [147, 20, 255], [255, 0, 0], [0, 0, 0], [0, 100, 0], [211,0,148], [0, 0, 255], [255, 255, 255], [147, 20, 255], [255, 0, 0], [0, 0, 0], [0, 100, 0], [211,0,148], [0, 0, 255]]
    # [white, pink, blue, black, green, purple, red]

    for kptno in range(len(left_kpts)):
        
        lpt = left_kpts[kptno]
        rpt = right_kpts[kptno]
        color = colors[kptno]

        for pno in range(7):
    
            lpoint = lpt[pno]
            rpoint = rpt[pno]
            rpoint[0] = rpoint[0]+width

            cv2.circle(img_pair, lpoint, 3, color, -1)
            cv2.circle(img_pair, rpoint, 3, color, -1)
            if line:
                start_point = (0, lpoint[1])
                end_point = (2*width, lpoint[1])
                cv2.line(img_pair, start_point, end_point, (255, 255, 0), 1)
        
        #hardcoded
        if kptno in [0, 5, 6]:
            pointorg = [lpoint[0]-75, lpoint[1]]
        else:
            pointorg = [lpoint[0]+20, lpoint[1]]
        
        if annots is not None:
            cv2.putText(img_pair, f'{str(kptno)}: {annots[kptno]}', pointorg, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img_pair, str(kptno), pointorg, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

    imshow(img_pair, wait=0)
    cv2.imwrite("Pair.png",img_pair)
    return img_pair

def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )
    
    # recurse
    shape = get_shape(lst[0], shape)

    return shape

def line_search(patch, strip, metric_type='mae', gray_yes=False, show=False, save_plt=None):
    
    '''
    
    '''

    assert patch.shape[0] == strip.shape[0]
    
    patch_copy = patch
    width = patch.shape[1]
    pad = int(width//2)

    padding = np.zeros((patch.shape[0], pad, patch.shape[2]))
    strip_pad = np.hstack((padding, strip, padding))

    # extracts = extract_patches_2d(strip_pad, patch.shape)
    if gray_yes:
        extracts = np.expand_dims(np.squeeze(view_as_windows(strip_pad, patch.shape)), axis=-1)
    else:
        extracts = np.squeeze(view_as_windows(strip_pad, patch.shape))
    patch = np.expand_dims(patch, axis=0)

    if metric_type == 'mae':
        metric_name = 'Mean Absolute Error'
        metric = np.sum(np.abs(extracts-patch), axis=(1,2,3))
   
    elif metric_type == 'rmse':
        metric_name = 'Root Mean Square Error'
        metric = np.sqrt(np.sum((extracts-patch)**2, axis=(1,2,3)))
   
    else:                                                                      

        patch_mean = np.expand_dims(np.mean(patch, axis=(1,2)), axis=(1,2))
        norm_patch = patch-patch_mean
        patch_std = np.expand_dims(np.std(patch, axis=(1,2)), axis=(1,2))

        extracts_mean = np.expand_dims(np.mean(extracts, axis=(1,2)), axis=(1,2)) 
        norm_extracts = extracts-extracts_mean                                    
        extracts_std = np.expand_dims(np.std(extracts, axis=(1,2)), axis=(1,2))   

        covars = np.expand_dims(np.mean(np.multiply(norm_patch, norm_extracts), axis=(1,2)), axis=(1,2))

        if metric_type == 'ncc':      
            metric_name = 'Normalized Cross-Correlation'
            
            numerator = np.mean(np.multiply(norm_patch, norm_extracts), axis=(1,2))
            denominator = np.squeeze(np.multiply(patch_std, extracts_std))

            metric = -np.mean(np.divide(numerator, denominator), axis=1)

        elif metric_type == 'ssim':
            metric_name = 'Structural Similarity'

            c1 = (0.01*255)**2
            c2 = (0.03*255)**2
        
            numerator = np.multiply(c1 + 2*np.multiply(patch_mean, extracts_mean), c2 + 2*covars)
            denominator = np.multiply((c1 + patch_mean**2 + extracts_mean**2), (c2 + patch_std**2 + extracts_std**2))

            if gray_yes:
                metric = -np.squeeze(np.divide(numerator, denominator))
            else:
                metric = -np.mean(np.squeeze(np.divide(numerator, denominator)), axis=1)

        # print(patch_mean.shape)
        # print(norm_patch.shape)
        # print(patch_std.shape)
        # print(extracts_mean.shape)
        # print(norm_extracts.shape)
        # print(extracts_std.shape)
        # print(covars.shape)
        # print(numerator.shape)
        # print(denominator.shape)
        # print(metric.shape)

    min_pt = np.argmin(metric)

    if show:

        imshow(patch_copy, 'patch from left image')
        imshow(strip, 'strip from right image')
        imshow(strip_pad, 'padding on strip')
            
        print('Strip shape: ', strip.shape)
        print('Padded strip shape: ', strip_pad.shape)
        print('Patch shape: ', patch.shape)
        print('Extracted patches shape: ', extracts.shape)  
        print('Metric array shape: ', metric.shape)
        print(f'Minimum value of metric at {min_pt}th location in strip!')

    if save_plt is not None:

        plt.figure(figsize=(12,7))
        plt.plot(np.arange(strip.shape[1]),metric,'b-')
        #plt.plot(np.arange(strip.shape[1]),metric,'ro')
        
        plt.title(f'{metric_name} vs. Distance on Scan Line (px)', fontsize=22)
        plt.xlabel('Distance on Scan Line (px)', fontsize=16)
        plt.ylabel(f'{metric_name}', fontsize=16)
        
        # plt.xticks(np.arange(0, 29, 1),fontsize=14) 
        # plt.yticks(np.arange(0, 16, 1), fontsize=14) 
        plt.savefig(save_plt)
        if show:
            plt.show()

    return int(min_pt)

def line_intersection(p_r1, p_r2, y_l):
    '''
    Does exactly what the name suggests
    TODO: Add support for tuples of y_l points
    '''

    x1, y1 = p_r1
    x2, y2 = p_r2
    
    m = (y2-y1)/(x2-x1)
    c = y1 - m*x1
    
    x_l = (y_l-c)/m
    return np.round(x_l).astype(np.int32)

def get_kpt_matches(img_l, img_r, l_kpts, patch_width = 5, disp_range = 16, metric='mae', gray_flag=False, show=False, verbose=False):

    '''
    WORKS ONLY LEFT KEYPOINTS TO RIGHT KEYPOINTS!!!!!!!!!!!!!!
    TODO: 
    > ASSUMES THAT THERE IS PATCH WIDTH SPACE AROUND KEYPOINT 
    > RESTRICT SEARCH SPACE IN X DIRECTION :DONE
    > APPLY CONSRAINT THAT EVERYTHING WILL BE LEFT SHIFTED: DONE
    '''

    r_kpts = []
    offset = int(patch_width//2)   
    width = img_l.shape[1]

    if gray_flag:
        img_l = np.expand_dims(gray(img_l), -1)
        img_r = np.expand_dims(gray(img_r), -1)

    def clip(t):
        return min(max(0, t), width-1)

    pnames = ['Apex', 'BL', 'BR']
    for idx, pt in enumerate(l_kpts):
        
        if idx>=0:
        
            l_triang = [pt[0], pt[5], pt[6]]
            r_triang = []

            for pname, l_pt in enumerate(l_triang):

                xm, ym = l_pt[0], l_pt[1]
                if(pname==0):
                    r_triang.append([0, ym])
                    continue
                patch = img_l[ym-offset:ym+offset+1, xm-offset:xm+offset+1]
                strip = img_r[ym-offset:ym+offset+1, clip(xm-disp_range):clip(int(xm+(disp_range//5))+1)]
                
                # if pnames[pname]=='Apex':
                #     if gray_flag:
                #         save_name = f'propagation_metrics/cone{idx}_{pnames[pname]}_{patch_width}_{disp_range}_{metric}_gray.png'
                #     else:
                #         save_name = f'propagation_metrics/cone{idx}_{pnames[pname]}_{patch_width}_{disp_range}_{metric}.png'
                # else:
                #     save_name = None

                save_name = None
                r_pt_x = clip(xm-disp_range)+line_search(patch, strip, metric_type=metric, gray_yes=gray_flag, save_plt=save_name)
                r_triang.append([r_pt_x, ym])

            x = [r_triang[1][0], r_triang[2][0]]
            y = [r_triang[1][1], r_triang[2][1]]
            # Approach 1 - finding apex using midpoint

            r_pt_x_apex = (r_triang[1][0]+r_triang[2][0])//2
            r_triang[0][0]=r_pt_x_apex    

            # Approach 2 - using euclidean distance

            # if (r_triang[1][1] == r_triang[2][1]):
            #     r_pt_x_apex = (r_triang[1][0]+r_triang[2][0])//2
            #     r_triang[0][0]=r_pt_x_apex  
            # else:
            #     apex_y = r_triang[0][1]
            #     apex_x = 0.5*((r_triang[1][0]+r_triang[2][0]) -((r_triang[1][1] - r_triang[2][1])*(2*apex_y-(r_triang[1][1]+r_triang[2][1])))) 
            #     r_triang[0][0] = apex_x
            
            # Approach 3 - finding slope and then solving 

            # slope, intercept, r_value, p_value, std_err = linregress(x, y)
            # if (slope == 0):
            #     r_pt_x_apex = (r_triang[1][0]+r_triang[2][0])//2
            #     r_triang[0][0]=r_pt_x_apex  
            # else:
            #     normal_slope = (-1/slope)
            #     print(normal_slope)   
            #     r_pt_x_midpoint = (r_triang[1][0]+r_triang[2][0])//2
            #     r_pt_y_midpoint = (r_triang[1][1]+r_triang[2][1])//2
            #     r_pt_y_apex = r_triang[0][1]
            #     r_pt_x_apex = r_pt_x_midpoint + (r_pt_y_apex - r_pt_y_midpoint)//normal_slope
            #     r_triang[0][0]=r_pt_x_apex

            xr1 = line_intersection(r_triang[0], r_triang[1], pt[1][1])
            xr2 = line_intersection(r_triang[0], r_triang[2], pt[2][1])
            xr3 = line_intersection(r_triang[0], r_triang[1], pt[3][1])
            xr4 = line_intersection(r_triang[0], r_triang[2], pt[4][1])

            rpt = [r_triang[0], [xr1, pt[1][1]], [xr2, pt[2][1]], [xr3, pt[3][1]], [xr4, pt[4][1]] , r_triang[1], r_triang[2]]
            
            if verbose:
                print(f'Left: {pt}\nRight: {rpt}\n')

            r_kpts.append(rpt)

    if show:
        img_pair =  draw_propagate(l_kpts, r_kpts, img_l, img_r, line=False)
        imshow(img_pair, caption = metric, wait=1)
        if gray_flag:
            cv2.imwrite(f'propagation_metrics/vis_{patch_width}_{disp_range}_{metric}_gray.png', img_pair)        
        else:
            cv2.imwrite(f'propagation_metrics/vis_{patch_width}_{disp_range}_{metric}.png', img_pair)
    
    return r_kpts

class PatchProcessor(Dataset):
    '''
    Dont't use Dataset
    '''
    def __init__(self, patches, strips, starts):
        """
        Args:
            patches (int): Array of patches from left image of shape (n_patches, ph, pw, n_channels)
            strips (int): Corresponding array of epipolar scan strips from right image of shape (n_patches, sh, sw, n_channels)
        """

        self.patches = torch.tensor(patches, dtype=torch.float).unsqueeze(-1)
        self.strips = torch.tensor(strips, dtype=torch.float)

        pad = int(self.patches.size()[1]//2)
        self.pstrips = F.pad(self.strips, (0, 0, pad, pad, 0, 0, 0, 0))
        self.pstrips = self.pstrips.unfold(dimension=2, size=self.patches.size()[1], step=1)
        
        self.pstrips = torch.permute(self.pstrips, (2, 0, 3, 1, 4))
        self.patches = torch.broadcast_to(torch.permute(self.patches, (4, 0, 3, 1, 2)), self.pstrips.size())

    def __len__(self):
        return self.pstrips.size()[0]

    def __getitem__(self, idx):
        return self.pstrips[idx], self.patches[idx]

class PatchMatcher():
    '''
    TODO: Implement SSIM, AE for our dataloader structure
    Works with only MAE, RMSE, PSNR
    '''
    
    def __init__(self, dataset, batch_size=16, sim_type='rmse'):
        
        if sim_type == 'mae':
            self.model = MAE()
        elif sim_type == 'rmse':
            self.model = RMSE()
        elif sim_type == 'ssim':
            self.model = SSIM()
        elif sim_type == 'psnr':
            self.model = PSNR()
        elif sim_type == 'ae':
            self.model = AE()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def __call__(self, l_keypoints, starts):
        '''
        Args:
            starts (int): Array of starting points of strips in left image of shape(n_patches, )
        '''
        self.starts = starts
        l_kpts = torch.tensor(l_keypoints, dtype=torch.float)
        for idx, (strips, patch) in enumerate(self.dataloader):
            
            strips, patch = strips.to(self.device), patch.to(self.device)

            metric = self.model(strips, patch)
            if idx==0:
                metrics=metric
            else:
                metrics = torch.cat((metrics, metric), dim=0)

        max_xs = torch.reshape(self.starts+torch.argmax(metrics, dim=0), (l_kpts.size()[0],-1))
        print(l_kpts)
        print(max_xs)

        return max_xs

if __name__ == '__main__':
    '''
    TODO: CHECK GRAYSCALE NCC IMPLEMENTATION
    '''
    
    print()
    with open ('lkpts.p', 'rb') as fp:
        l_kpts = pickle.load(fp)
        
    # with open('bboxes.p', 'wb') as fp:
    #     pickle.dump(left_boxes, fp)
    # with open('lkpts.p', 'wb') as fp:
    #     pickle.dump(left_kpts, fp)
    # with open ('bboxes.p', 'rb') as fp:
    #     left_boxes = pickle.load(fp) 

    # line_search(np.zeros((33,33,3))+5, np.ones((33,483,3)), metric_type='ncc', show=True)
    # line_search(np.random.randn(33,33,3)+5, np.random.randn(33,483,3), metric_type='ncc', show=True)
    # for metrics in ['mae', 'rmse', 'ncc', 'ssim']:
        #   for flg in [True, False]:
            #     get_kpt_matches(left_image, right_image, l_kpts, patch_width=32, disp_range=256, metric=metrics, gray_flag=flg, show=True)    
    
    """
    left_img_path = "stereo_image/left_image.jpeg"
    right_img_path = "stereo_image/right_image.jpeg"

    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)
    with open ('lkpts.p', 'rb') as fp:
        l_kpts = pickle.load(fp)
        
    patch_width = 32
    disp_range = 256
    gray_flag=False

    r_kpts = []
    offset = int(patch_width//2)   
    width = img_l.shape[1]

    if gray_flag:
        img_l = np.expand_dims(gray(img_l), -1)
        img_r = np.expand_dims(gray(img_r), -1)

    def clip(t):
        return min(max(0, t), width-1)

    count=0
    for idx, pt in enumerate(l_kpts):
            
        l_triang = [pt[0], pt[5], pt[6]]
        r_triang = []

        for pno, l_pt in enumerate(l_triang):
            
            xm, ym = l_pt[0], l_pt[1]

            strip_l = clip(xm-disp_range)                
            strip_r = clip(int(xm+(disp_range//5)+1))

            patch = np.expand_dims(img_l[ym-offset:ym+offset+1, xm-offset:xm+offset+1], axis=0)
            strip = np.expand_dims(img_r[ym-offset:ym+offset+1, strip_l:strip_r], axis=0)

            if strip_l == 0:
                strip = np.pad(strip, ((0,0),(0,0),(-(xm-disp_range),0),(0,0)))
            if strip_r == 0:
                strip = np.pad(strip, ((0,0),(0,0),(0,int(xm+(disp_range//5)+1)-(width-1)),(0,0)))

            if count == 0:
                patches = patch
                strips = strip
                starts = [strip_l]
            else:
                patches = np.concatenate([patches, patch], axis=0)
                strips = np.concatenate([strips, strip], axis=0)
                starts.append(strip_l)

            count += 1
    starts = torch.tensor(starts)


    dataset = PatchProcessor(patches, strips, starts)
    matcher = PatchMatcher(dataset, batch_size=32, sim_type='rmse')
    r_kpts = matcher(l_kpts, starts)

    print(r_kpts.shape)
    """