import torch
import cv2
import pickle
import numpy as np

class DepthFinder:

	def __init__(self, focus =392, baseline = 170, pixel_size = 0.004):
		#focus 1000 for 1280x720 zed images (old) and baseline 120 for zed
		#baseline 320 for fsds and focus 392
		#

		"""
		Using ZED 2 HD1080 default parameters:
		https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
		:param focus: Focal length of the camera in mm - 2000px
		:param baseline: Baseline distance of the cameras in mm - 12 cm
		:param pixel_size: Size of pixel for the camera in mm - 2 microns/px
		"""
		self.focus = focus
		self.baseline = baseline
		self.pixel_size = pixel_size
		self.prod = torch.tensor(self.focus*self.baseline/1000)
		# print(f'Depth(m) * Disparity(px) = {self.focus*self.baseline/1000} m.px')

	def find_depth(self, left_kps, right_kps):
		'''
		Input: kps: torch tensors of size (n_cones, 7, 2)
		'''
		global depth_mean
		
		if left_kps.numel()==0:
			pass
		else:
			disparity = (left_kps-right_kps)[:,0]
			depths = torch.abs(torch.divide(self.prod, disparity))
			depth_mean = torch.mean(depths)
		
		return depth_mean.item()

class Triangulation:
    
	focus = 0
	baseline = 0
	pixel_size = 0

	def __init__(self, focus =392, baseline = 320, pixel_size = 0.004):
		"""
		:param focus: Focal length of the camera in mm
		:param baseline: Baseline distance of the cameras in mm
		:param pixel_size: Size of pixel for the camera
		"""
		self.focus = focus
		self.baseline = baseline
		self.pixel_size = pixel_size

	def find_depth(self, left_points, right_points):
		"""
		Function to calculate depth of cone by applying Linear regression on disparity values for given points
		:param left_points: n*2 tensor containing pixel coordinates of feature points from left image
		:param right_points: n*2 tensor containing pixel coordinates of feature points from right image
		:return: depth
		"""

		if self.focus == 0 or self.baseline == 0:
			raise AssertionError("Error: Focus or Baseline not valid")

		disparities = abs(left_points - right_points)[:,0].clone().detach().requires_grad_(False).reshape(-1,1)
		y_val = torch.ones(disparities.size()) * (self.baseline * self.focus/self.pixel_size)

		# Apply Linear Regression
		depth = 1e6*torch.median(torch.div(disparities, y_val))    # B*f/pixel_size (y) = depth (m) * disparity (x)
		return depth

if __name__ == '__main__':

	# print(torch.tensor([[85, 451], [80, 463], [92, 463], [79, 473], [94, 472], [75, 484], [98, 482]]).shape)
	# left = cv2.imread('D:/Python Projects/Stereo/Left_Right_Middle/image_left_18.png')
	# right = cv2.imread('D:/Python Projects/Stereo/Left_Right_Middle/image_right_18.png')
	#
	# for idx, point in enumerate([[85,451],[80,463],[92,463],[79,473],[94,472],[75,484],[98,482]]):
	# 	# coords = pd.eval(point)
	# 	coords = point
	# 	cv2.circle(left, coords, 2, [0,255,0], -1)
	#
	# for idx, point in enumerate([[114,440],[109,447],[118,447],[108,456],[119,455],[105,464],[121,463]]):
	# 	# coords = pd.eval(point)
	# 	coords = point
	# 	cv2.circle(right, coords, 2, [0, 0, 255], -1)
	#
	# cv2.imshow('left', left)
	# cv2.imshow('right', right)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# tringl = Triangulation(26, 5, 0.0008)
	# l = torch.tensor([(443, 961), (465, 1362), (534, 907), (539, 906), (562, 955), (608, 1349), (633, 1002), (645, 1369), (670, 1026), (672, 1029), (688, 1313), (691, 1322)], dtype=torch.float32)
	# r = torch.tensor([(955, 1492), (1029, 1081), (1079, 1079), (1084, 1076), (951, 1212), (1147, 1526), (1225, 1122), (1029, 1081), (1029, 1081), (1221, 1412), (1199, 1488), (1170, 1364)], dtype=torch.float32)
	# # print(l.dtype, r.dtype)
	# depth = tringl.find_depth(l,r)
	# print(depth)

	depthfinder = DepthFinder()
	with open ('lkpts.p', 'rb') as fp:
		l_kpts = pickle.load(fp)
	depth = depthfinder.find_depth(torch.tensor([l_kpts[0], l_kpts[1]]), torch.tensor([l_kpts[1], l_kpts[0]]))
	print(depth, depth.size())