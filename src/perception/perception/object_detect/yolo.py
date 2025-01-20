import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append("object_detect/yolov5/")

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, strip_optimizer, scale_segments, xyxy2xywh)  
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync


class Yolo():
	def __init__(self,
				 weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
				 data=ROOT / 'models/data.yaml',  # dataset.yaml path
				 imgsz=(640, 640),  # inference size (height, width)
				 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
				 half=False,  # use FP16 half-precision inference
				 dnn=False,  # use OpenCV DNN for ONNX inference
				 ):

		# Load model
		self.weights = weights
		self.device = select_device(device)
		self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
		self.stride, self.names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
		self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

		# Half
		half &= (self.pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
		if self.pt or jit:
			self.model.model.half() if half else self.model.model.float()
		self.half = half

	def detect_all(self,
			   source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
			   name='exp',  # save results to project/name
			   exist_ok=False,  # existing project/name ok, do not increment
			   project=ROOT / 'runs/detect',  # save results to project/name
			   conf_thres=0.25,  # confidence threshold
			   iou_thres=0.45,  # NMS IOU threshold
			   max_det=1000,  # maximum detections per image
			   view_img=False,  # show results
			   save_txt=False,  # save results to *.txt
			   save_conf=False,  # save confidences in --save-txt labels
			   save_crop=False,  # save cropped prediction boxes
			   nosave=False,  # do not save images/videos
			   classes=None,  # filter by class: --class 0, or --class 0 2 3
			   agnostic_nms=False,  # class-agnostic NMS
			   augment=False,  # augmented inference
			   visualize=False,  # visualize features
			   update=False,  # update all models
			   line_thickness=3,  # bounding box thickness (pixels)
			   hide_labels=False,  # hide labels
			   hide_conf=False,  # hide confidences
			   ):

		source = str(source)
		save_img = not nosave and not source.endswith('.txt')  # save inference images
		is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
		# is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
		webcam = source.isnumeric() or source.endswith('.txt') or (not is_file)
		if is_file:
			source = check_file(source)  # download

		# Directories
		#save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
		#save_dir = increment_path(Path(project) , exist_ok=exist_ok)  # increment run
		save_dir = Path(project)
		(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

		# Dataloader
		if webcam:
			view_img = check_imshow()
			cudnn.benchmark = True  # set True to speed up constant image size inference
			dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
			bs = len(dataset)  # batch_size
		else:
			dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
			bs = 1  # batch_size
		vid_path, vid_writer = [None] * bs, [None] * bs

		self.model.warmup(imgsz=(1 if self.pt else bs, 3, *(self.imgsz)))  # warmup
		dt, seen = [0.0, 0.0, 0.0], 0

		all_boxes = []
		for path, im, im0s, vid_cap, s in dataset:
			t1 = time_sync()
			im = torch.from_numpy(im).to(self.device)
			im = im.half() if self.half else im.float()  # uint8 to fp16/32
			im /= 255  # 0 - 255 to 0.0 - 1.0
			if len(im.shape) == 3:
				im = im[None]  # expand for batch dim
			t2 = time_sync()
			dt[0] += t2 - t1

			# Inference
			visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
			pred = self.model(im, augment=augment, visualize=visualize)
			t3 = time_sync()
			dt[1] += t3 - t2

			# NMS
			pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
			dt[2] += time_sync() - t3

			# Second-stage classifier (optional)
			# pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

			# Process predictions
			img_preds = []
			for i, det in enumerate(pred):  # per image
				seen += 1
				if webcam:  # batch_size >= 1
					p, im0, frame = path[i], im0s[i].copy(), dataset.count
					s += f'{i}: '
				else:
					p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

				p = Path(p)  # to Path
				save_path = str(save_dir / p.name)  # im.jpg
				txt_path = str(save_dir / 'labels' / p.stem) + (
					'' if dataset.mode == 'image' else f'_{frame}')  # im.txt
				s += '%gx%g ' % im.shape[2:]  # print string
				gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
				imc = im0.copy() if save_crop else im0  # for save_crop
				annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
				if len(det):
					# Rescale boxes from img_size to im0 size
					det[:, :4] = scale_segments(im.shape[2:], det[:, :4], im0.shape).round()

					# Print results
					for c in det[:, -1].unique():
						n = (det[:, -1] == c).sum()  # detections per class
						s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

					# Write results
					for *xyxy, conf, cls in reversed(det):
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						img_preds.append([cls.item(), xywh, conf.item()])
						if save_txt:  # Write to file
							# xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
							line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
							with open(txt_path + '.txt', 'a') as f:
								f.write(('%g ' * len(line)).rstrip() % line + '\n')

						if save_img or save_crop or view_img:  # Add bbox to image
							c = int(cls)  # integer class
							label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
							annotator.box_label(xyxy, label, color=colors(c, True))
							if save_crop:
								save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

				# Stream results
				im0 = annotator.result()
				if view_img:
					cv2.imshow(str(p), im0)
					cv2.waitKey(1)  # 1 millisecond

				# Save results (image with detections)
				if save_img:
					if dataset.mode == 'image':
						cv2.imwrite(save_path, im0)
					else:  # 'video' or 'stream'
						if vid_path[i] != save_path:  # new video
							vid_path[i] = save_path
							if isinstance(vid_writer[i], cv2.VideoWriter):
								vid_writer[i].release()  # release previous video writer
							if vid_cap:  # video
								fps = vid_cap.get(cv2.CAP_PROP_FPS)
								w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
								h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
							else:  # stream
								fps, w, h = 30, im0.shape[1], im0.shape[0]
							save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
							vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
						vid_writer[i].write(im0)

			# Print time (inference-only)
			LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
			all_boxes.append(img_preds)

		# Print results
		t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
		LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *(self.imgsz))}' % t)
		if save_txt or save_img:
			s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
			LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
		if update:
			strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)
		return all_boxes

	def warmup(self):
		self.model.warmup(imgsz=(1 , 3, *(self.imgsz)), half=self.half)

	def detect_cont(self,
					img, # Frame to be detected
					name='exp',  # save results to project/name
					exist_ok=False,  # existing project/name ok, do not increment
					project=ROOT / 'runs/detect',  # save results to project/name
					augment=False,  # augmented inference
			   		visualize=False,  # visualize features
					conf_thres=0.25,  # confidence threshold
					iou_thres=0.45,  # NMS IOU threshold
					max_det=1000,  # maximum detections per image
					view_img=False,  # show results
					save_txt=False,  # save results to *.txt
					save_conf=False,  # save confidences in --save-txt labels
					save_crop=False,  # save cropped prediction boxes
					nosave=False,  # do not save images/videos
					classes=None,  # filter by class: --class 0, or --class 0 2 3
					agnostic_nms=False,  # class-agnostic NMS
					line_thickness=3,  # bounding box thickness (pixels)
					hide_labels=False,  # hide labels
					hide_conf=False,  # hide confidences
					):

		from yolov5.utils.augmentations import letterbox
		import numpy as np

		save_img = not nosave  # save inference image
		# Directories
		save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
		(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

		# Preprocess Img
		assert img is not None, f'Image not received from camera'
		# Padded resize
		img0 = letterbox(img, self.img_size, stride=self.stride, auto=self.auto)[0]

		# Convert
		img0 = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
		img0 = np.ascontiguousarray(img0)

		dt, seen = [0.0,0.0,0.0], 0
		t1 = time_sync()
		img = torch.from_numpy(img0).to(self.device)
		img = img.half() if self.half else img.float()  # uint8 to fp16/32
		img /= 255  # 0 - 255 to 0.0 - 1.0
		if len(img.shape) == 3:
			img = img[None]  # expand for batch dim
		t2 = time_sync()
		dt[0] += t2 - t1

		# Inference
		# visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
		pred = self.model(img, augment=augment, visualize=visualize)
		t3 = time_sync()
		dt[1] += t3 - t2

		# NMS
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
		dt[2] += time_sync() - t3

		# Second-stage classifier (optional)
		# pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

		# Process predictions
		s = ""
		boxes = []
		for i, det in enumerate(pred):  # per image
			seen += 1
			im0 = img0.copy()

			save_path = str(save_dir / 'detect')  # im.jpg
			txt_path = str(save_dir / 'labels' / 'detect') # im.txt
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			imc = im0.copy() if save_crop else im0  # for save_crop
			annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_segments(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
					boxes.append((cls, xywh, conf))

					if save_txt:  # Write to file
						line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * len(line)).rstrip() % line + '\n')

					if save_img or save_crop or view_img:  # Add bbox to image
						c = int(cls)  # integer class
						label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
						annotator.box_label(xyxy, label, color=colors(c, True))
						if save_crop:
							save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'detect.jpg', BGR=True)

			# Stream results
			im0 = annotator.result()
			if view_img:
				cv2.imshow(str("Cone Detections"), im0)
				cv2.waitKey(1)  # 1 millisecond

			# Save results (image with detections)
			if save_img:
				cv2.imwrite(save_path, im0)

		# Print time (inference-only)
		LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
		return boxes


