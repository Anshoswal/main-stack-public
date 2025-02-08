import torch
import numpy as np
import cv2


class Features:

	def __init__(self):
		self.sift = cv2.SIFT_create()
		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		self.flann = cv2.FlannBasedMatcher(index_params,search_params)
		self.bf=cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

	def feature_detect(self, left_boxes, right_boxes, left_img, right_img, draw=0):
		"""
		Extract feature points from the left and right images for all the provided bounding boxes using SIFT algorithm

		:param right_img: Image from right camera
		:param left_img: Image from left camera
		:param left_boxes: Bounding Box params for left image
		:param right_boxes:	Bounding Box params propagated on right image
		:param draw: Boolean variable. If 1, draws the feature points and prints the image on the screen.
		:return: Feature keypoints and descriptors for left and right image

		"""
		# Grayscale
		#gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
		#edged_left = cv2.Canny(gray_left, 30, 200)

		# gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
		# edged_right = cv2.Canny(gray_right, 30, 200)

		self.left_image = left_img
		self.right_image = right_img

		# Left Boxes
		left_keypoints = []
		left_descriptors = []
		for left_box in left_boxes:
			x, y, w, h = left_box
			lx= self.left_image.shape[1]
			ly=self.left_image.shape[0]
			"""
			x1 = int((x - w / 2) * lx)
			y1 = int((y - h / 2) * ly)
			x2 = int((x + w / 2) * lx)
			y2 = int((y + h / 2) * ly)
			"""
			x1 = int((x - w / 7) * lx)
			y1 = int((y - h/3 ) * ly)
			x2 = int((x + w / 7) * lx)
			y2 = int((y + h / 2.1) * ly)
			
			
			
			
			mask = np.zeros(self.left_image.shape[:2], dtype = 'uint8')
			mask[y1:y2,x1:x2] = np.ones(mask[y1:y2,x1:x2].shape,dtype=np.uint8)*255

			keypoints, descriptors = self.sift.detectAndCompute(self.left_image, mask=mask)
			left_keypoints.append(keypoints)
			left_descriptors.append(descriptors)

		# Right Box
		right_keypoints = []
		right_descriptors = []
		for right_box in right_boxes:
			x, y, w, h = right_box
			"""
			x1 = int((x - w / 2) )
			y1 = int((y - h / 2) )
			x2 = int((x + w / 2) )
			y2 = int((y + h / 2) )
			"""
			x1 = int((x - w / 7) )
			y1 = int((y - h/3 ) )
			x2 = int((x + w / 7) )
			y2 = int((y + h / 2.1) )
			
			
			
			mask = np.zeros(self.right_image.shape[:2], dtype='uint8')
			mask[y1:y2, x1:x2] = np.ones(mask[y1:y2, x1:x2].shape, dtype=np.uint8) * 255

			keypoints, descriptors = self.sift.detectAndCompute(self.right_image, mask=mask)
			right_keypoints.append(keypoints)
			right_descriptors.append(descriptors)

		if draw == 1:
			sift_image = self.left_image
			for points in left_keypoints:
				sift_image = cv2.drawKeypoints(sift_image, points, outImage=None)
			sift_image = cv2.resize(sift_image,(1280,720))
			cv2.imshow("image", sift_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		if draw == 1:
			sift_image = self.right_image
			for points in right_keypoints:
				sift_image = cv2.drawKeypoints(sift_image, points, outImage=None)
			sift_image = cv2.resize(sift_image,(1280,720))
			cv2.imshow("image", sift_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		features = {"keypoints": [left_keypoints, right_keypoints],
					"descriptors": [left_descriptors, right_descriptors]}

		return features

	def feature_matching(self, features,left_bboxes,right_bboxes,l_kpts_rektnet, draw=0):
		"""
		Match feature points using Lowe's ratio test and return matched keypoints in left and right image for each bbox

		:param features: Dict containing left and right feature keypoints and descriptors for each box in frame.
		:param draw: Boolean variable. If 1, draws the matched feature points and prints the image on the screen.
		:return keypoints: keypoints["left"][i] will store coordinates of all the feature points for box i in left frame (as floats)
		"""
		

		left_descriptor = features["descriptors"][0]
		right_descriptor = features["descriptors"][1]
		left_keypoints = features["keypoints"][0]
		right_keypoints = features["keypoints"][1]
		#print(left_keypoints)

		all_matches = []		# stores all cv2::DMatch (from left to right) objects for each box in frame
		
		left_keypoints_new=[]
		right_keypoints_new=[]
		left_descriptor_new=[]
		right_descriptor_new=[]
		left_boxes_new=[]
		right_boxes_new=[]
		l_kpts_rektnet_new=[]


		for i in range(len(left_keypoints)):

			# knn_matches = self.flann.knnMatch(cv2.UMat(left_descriptor[i]), cv2.UMat(right_descriptor[i]), 2)
			# ratio_thresh = 0.7
			# good_matches = []		# matches for box i
			# for m, n in knn_matches:
			# 	if m.distance < ratio_thresh * n.distance:
			# 		good_matches.append(m)
			# all_matches.append(good_matches)

			#print("LEFT KPT ", len(left_keypoints[i]), "RIGHT KPT ",len(right_keypoints[i]))
			try:
				if len(left_keypoints[i]) != 0 :
					matches = self.bf.match(cv2.UMat(left_descriptor[i]),cv2.UMat(right_descriptor[i]))
					matches = sorted(matches, key = lambda x:x.distance)
					matches=matches[:1]
					all_matches.append(matches)

					#the following is to make sure only non empty keypoints are retained
					left_keypoints_new.append(left_keypoints[i])
					left_descriptor_new.append(left_descriptor[i])
					right_keypoints_new.append(right_keypoints[i])
					right_descriptor_new.append(right_descriptor[i])
					left_boxes_new.append(left_bboxes[i])
					right_boxes_new.append(right_bboxes[i])
					l_kpts_rektnet_new.append(l_kpts_rektnet[i])



			except cv2.error:
				print("ERROR IN DEPTH ESTIMATION OF CONE due to no kpt found, MOVING ON")

				#remove cones in which no keypoints can be found

		left_keypoints=left_keypoints_new
		left_descriptor=left_descriptor_new
		right_keypoints=right_keypoints_new
		right_descriptor=right_descriptor_new
			

		

		# Get KeyPoint Coordinates
		keypoints = {"left": [], "right": []}
		for i in range(len(all_matches)):
			box_matches = all_matches[i]
			box_kp_l = left_keypoints[i]
			box_kp_r = right_keypoints[i]
			pts_l = []
			pts_r = []
			for match in box_matches:
				point_l = tuple(map(int,box_kp_l[match.queryIdx].pt))
				point_r = tuple(map(int,box_kp_r[match.trainIdx].pt))
				
				pts_l.append(point_l)
				pts_r.append(point_r)


			# keypoints["left"][i] will store coordinates of all the feature points for box i in left frame (as floats)
			keypoints["left"].append(pts_l)
			keypoints["right"].append(pts_r)
		#print(keypoints)
		# Visualization and Debugging
		if draw == 1:
			
			for i in range(len(all_matches)):
				img_matches = np.empty((max(self.left_image.shape[0], self.right_image.shape[0]),self.left_image.shape[1] + self.right_image.shape[1], 3), dtype=np.uint8)
				cv2.drawMatches(self.left_image, left_keypoints[i], self.right_image, right_keypoints[i], all_matches[i],img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
								
				x=int(2*self.right_image.shape[1]*0.4)
				y=int(self.right_image.shape[0]*0.4)
				img_matches = cv2.resize(img_matches,(x,y),interpolation=cv2.INTER_AREA)
				cv2.imshow('Good Matches', img_matches)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		#print("LEFT KPTS FROM SIFT  ",keypoints["left"], "\n" ,"RIGHT KPTS FROM SIFT ", keypoints["right"])
		return keypoints["left"],keypoints["right"],left_boxes_new,right_boxes_new,l_kpts_rektnet_new
	

if __name__ == '__main__':


	#WONT RUN FROM MAIN BECAUSE IN LEFT AND RIGHT BOXES WALA PART, LEFT BOXES ARE MULTIPLIED BY
	#lx ly etc but RIGHT BOXES ARE NOT MULTIPLIED BY THAT BECAUSE THEY ARE MADE TO RUN BY TAKING DATA FROM BB PROPAGATION BY RUNNING DISPARITY



	left_image = cv2.imread('D:/COLLEGE/racing/V4-sift_works/stereoPerception-master/stereoPerception-master/camera_data/left_image5.png')
	right_image = cv2.imread('D:/COLLEGE/racing/V4-sift_works/stereoPerception-master/stereoPerception-master/camera_data/right_image5.png')
	xyz = Features()
	"""
	right_boxes=[[0.6050955653190613, 0.5420382022857666, 0.012738853693008423, 0.02165605127811432]]
	left_boxes=[[0.5847133994102478, 0.5414012670516968, 0.012738853693008423, 0.020382165908813477]]
	"""
	left_boxes_old= [[2.0, [0.560546875, 0.6465277671813965, 0.02109375037252903, 0.05694444477558136], 0.8945415019989014], 
                	[3.0, [0.326171875, 0.6388888955116272, 0.01953125, 0.06111111119389534], 0.9019558429718018], 
					[0.0, [0.5902343988418579, 0.6611111164093018, 0.02421874925494194, 0.06388889253139496], 0.9113948345184326], 
					[0.0, [0.48945313692092896, 0.6402778029441833, 0.01953125, 0.05277777835726738], 0.9145551323890686], 
					[0.0, [0.4488281309604645, 0.7090277671813965, 0.03515625, 0.09305555373430252], 0.9212427139282227], 
					[3.0, [0.25312501192092896, 0.7138888835906982, 0.03593749925494194, 0.10000000149011612], 0.9225156307220459], 
					[3.0, [0.39140623807907104, 0.6583333611488342, 0.02500000037252903, 0.0694444477558136], 0.9257155656814575], 
					[2.0, [0.614453136920929, 0.731249988079071, 0.03984374925494194, 0.11249999701976776], 0.9308196306228638], 
					[3.0, [0.33867186307907104, 0.831250011920929, 0.06015624850988388, 0.16527777910232544], 0.9324220418930054], 
					[0.0, [0.6675781011581421, 0.8416666388511658, 0.06484375149011612, 0.17499999701976776], 0.9346836805343628]]

	right_boxes_old=[[2.0, [0.5484374761581421, 0.6465277671813965, 0.02187499962747097, 0.05694444477558136], 0.8928252458572388], 
					[3.0, [0.314453125, 0.6388888955116272, 0.02109375037252903, 0.0555555559694767], 0.8959217667579651], 
					[0.0, [0.575390636920929, 0.6618055701255798, 0.02265625074505806, 0.06527777761220932], 0.9031292796134949],
					[0.0, [0.4781250059604645, 0.6402778029441833, 0.02031249925494194, 0.05277777835726738], 0.9153380393981934],
					[0.0, [0.4292968809604645, 0.7090277671813965, 0.03671874850988388, 0.09583333134651184], 0.9324054718017578],
					[3.0, [0.23281249403953552, 0.7124999761581421, 0.03593749925494194, 0.0972222238779068], 0.9235743284225464],
					[3.0, [0.3765625059604645, 0.6590277552604675, 0.02656250074505806, 0.06805555522441864], 0.9088550209999084],
					[2.0, [0.5921875238418579, 0.7291666865348816, 0.03750000149011612, 0.1111111119389534], 0.8176341652870178],
					[3.0, [0.30546873807907104, 0.831944465637207, 0.06406249850988388, 0.1666666716337204], 0.9396976828575134],
					[0.0, [0.633593738079071, 0.8409722447395325, 0.06406249850988388, 0.1736111044883728], 0.9317039251327515]]
	left_boxes=[]
	right_boxes=[]

	for cls,xywh,conf in left_boxes_old:
		left_boxes.append(xywh)
	for cls,xywh,conf in right_boxes_old:
		right_boxes.append(xywh)
	
	feat = xyz.feature_detect(left_boxes, right_boxes, left_image, right_image, draw=1)
	kpts = xyz.feature_matching(feat, draw=1)