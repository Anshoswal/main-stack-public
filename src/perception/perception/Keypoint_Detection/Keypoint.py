import sys
sys.path.append('Keypoint_Detection/')

import cv2
import numpy as np
import torch
import os
#import onnx
#import onnx_tensorrt.backend as backend

from perception.Keypoint_Detection.RektNet.keypoint_net import KeypointNet
from perception.Keypoint_Detection.RektNet.utils import prep_image

def flat_softmax(inp):
    flat = inp.view(-1, 80 * 80)
    flat = torch.nn.functional.softmax(flat, 1)
    return flat.view(-1, 7, 80, 80)

def soft_argmax(inp):
    values_y = torch.linspace(0, (80 - 1) / 80, 80, dtype=inp.dtype, device=inp.device)
    values_x = torch.linspace(0, (80 - 1) / 80, 80, dtype=inp.dtype, device=inp.device)
    exp_y = (inp.sum(3) * values_y).sum(-1)
    exp_x = (inp.sum(2) * values_x).sum(-1)
    return torch.stack([exp_x, exp_y], -1)

class Keypoints:

	def __init__(self, model_path):
		if model_path[-1] == 't':
			self.model = KeypointNet()
			self.model.load_state_dict(torch.load(model_path).get('model'))
			self.model.eval()
			
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			self.model.to(device)
			self.trt = False
		#the following elif is irrelavent to use since we aint gonna be using onnx
		elif model_path[-1] == 'x':
			self.model = onnx.load(model_path)
			self.model = backend.prepare(self.model,device='Cuda:1')
			self.trt = True

	def get_keypoints(self, image, vis = False):
		h, w, _ = image.shape
		image = prep_image(image=image, target_image_size=(80,80))
		image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]		# [H,W,C] => [C,H,W]

		if self.trt:
			image = image.astype('float32')
			image = np.array(image, dtype=image.dtype, order='C')
			hm = self.model.run(image)[0]
			hm = torch.from_numpy(hm).type('torch.FloatTensor')
			hm = flat_softmax(hm)
			out = KeypointNet.soft_argmax(hm)
			kpt = out.view(-1, 7, 2)

		else:
			image = torch.from_numpy(image).type('torch.FloatTensor')
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			image=image.to(device)
			[hm, kpt] = self.model(image)	# kpt = [7,x,y]
		if(vis):
			out = np.empty(shape=(0, hm[0].shape[2]))  # Out is the image with the points marked
			for o in hm[0]:
				chan = np.array(o.cpu().data)
				cmin = chan.min()
				cmax = chan.max()
				chan -= cmin
				chan /= cmax - cmin
				out = np.concatenate((out, chan), axis=0)

			# TODO: Add paths for output image
			output_path = " "
			img_name = " "
			print(cv2.imwrite(os.path.join(output_path, str(img_name + "_hm.jpg")), out * 255))
			print(f'please check the output image here: {output_path + img_name + "_hm.jpg", out * 255}')
			print(
				"-------------------------------------------------------------------------------------------------------------------------------------")
			print('\nPlease check the output image here: ',
				  (output_path + '\\' + str(img_name.split("\\")[-1]) + "_inference.jpg"))

		return kpt.detach().cpu().numpy().squeeze()		# returns the 7 points for one cone


#the following is useless for us since we never run Keypoint.py directly, we always call in in some other place (disparity.py in our case)
if __name__ == "__main__":
	mykpt = Keypoints("C:/Users/rampu/Desktop/racing/new_stereo/perception-master/Keypoint_Detection/Weights/23_loss_0.38.pt")
	image = cv2.imread('../stereo_image/yolov5fsdstest/content/yolov5/runs/detect/exp/crops/2/left.jpg')
	kpts = mykpt.get_keypoints(image)
	h,w,_ = image.shape
	# print(kpts)
	# x = kpts[0][0]
	kpts = kpts * [[w,h]]
	print(kpts)
	for pt in kpts:
		cvpt = (int(pt[0]), int(pt[1]))
		cv2.circle(image, cvpt, 3, (0, 255, 0), -1)
	# image = cv2.resize(image, fx = 0.5, fy = 0.5)
	cv2.imshow("out",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()