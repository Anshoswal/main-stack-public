# from tkinter import W
import cv2
import numpy as np

def propagate(left_pts, left_img, right_img, radius = 2, draw = 0):
	sift = cv2.SIFT_create()
	right_pts = []
	for pt in left_pts:
		x,y = pt
		left_kpt = cv2.KeyPoint(x,y,radius)
		right_kpts = []
		length = right_img.shape[0]
		for i in range(1, length, radius):
			temp = cv2.KeyPoint(i, y, radius)
			right_kpts.append(temp)

		kp, rdesc = sift.compute(right_img, right_kpts)
		kp, ldesc = sift.compute(left_img, [left_kpt])
		bf = cv2.BFMatcher()
		matches = bf.match(ldesc, rdesc)
		matches = sorted(matches, key=lambda a: a.distance)

		for match in matches:
			idx = match.trainIdx
			(xr,yr) = right_kpts[idx].pt
			right_pts.append((int(xr),int(yr)))

	if draw == 1:
		imgout = np.concatenate((left_img, right_img), axis=1)
		for i in range(len(right_pts)):
			color = tuple(np.random.randint(0,255,3))
			color = (int(color[0]), int(color[1]), int(color[2]))
			imgout = cv2.circle(imgout, left_pts[i], 3, color, 1)
			right = (int(right_pts[i][0]+left_img.shape[0]), int(right_pts[i][1]))
			imgout = cv2.circle(imgout, right, 3, color, 1)
			imgout = cv2.line(imgout, left_pts[i], right, color, 1)

		cv2.imshow("Matched points", imgout)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return right_pts

def get_bbox_from_kpts(kpts,left_boxes, img = None, draw = 0):
	boxes=[]
	for i in range(len(kpts)):
		box=[]
		cone_kpts= kpts[i]
		cone_kpts = sorted(cone_kpts, key = lambda pts: pts[0])
		x = np.mean(cone_kpts,axis=0,dtype=np.float32)[0]
		w = left_boxes[i][1][2]
		h = left_boxes[i][1][3]
		w = w*img.shape[1]
		h = h*img.shape[0]
		y = left_boxes[i][1][1] 
		y = y*img.shape[0]
		y = y-h/2
		x = x-w/2
		start_x=int(x)
		start_y=int(y)
		y=y+h
		x=x+w
		end_x=int(x)
		end_y=int(y)

		color = tuple(np.random.randint(0, 255, 3))
		color = (int(color[0]), int(color[1]), int(color[2]))
		out_img = cv2.rectangle(img, (start_x,start_y), (end_x,end_y), color, 3) 
		box.append(x-w/2)
		box.append(y-h/2)
		w = int(w)
		h = int(h)
		box.append(w)
		box.append(h)
		boxes.append(box)
	
	if draw==1:
		cv2.imshow("Matched points kpt", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return boxes
	

def draw_bbox(bb, img): 
	xr,yr,w,h = bb   
	xr = int(xr-w/2)
	yr = int(yr-h/2)
	w = int(w)
	h = int(h)
	color = tuple(np.random.randint(0, 255, 3))
	color = (int(color[0]), int(color[1]), int(color[2]))
	out_img = cv2.rectangle(img,(int(xr),int(yr)), (int(xr+w),int(yr+h)), color, 3)
	cv2.imshow("Matched points bb", out_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return (xr,yr,w,h)


if __name__ == "__main__":
	left_image = cv2.imread('../stereo_image/left_image.jpeg')
	right_image = cv2.imread('../stereo_image/right_image.jpeg')

	left_pts = []
	pts = propagate(left_pts, left_image, right_image, draw=0)
	get_bbox_from_kpts(pts, right_image, 1)