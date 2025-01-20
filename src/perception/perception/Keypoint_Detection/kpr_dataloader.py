# data loader to send images of bounding boxes all together instead of one by one to kpr



from distutils.command.build_scripts import first_line_re
from turtle import left
import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas
import cv2

class CustomDataset(Dataset):

    def __init__(self,left_boxes,left_image):
        #left_boxes_array=np.array(left_boxes)
        #self.x=left_boxes_array #x is inputs ie the cls,xywh,conf of bbs

        #self.y=np.array([]) #y is the output ie bounding box cropped images

        #print(self.x, type(self.x))
        #print(left_boxes_points)
            
        self.x = left_boxes #x is inputs ie the cls,xywh,conf of bbs
       
        self.n_samples=len(self.x)

        self.image=left_image
        device=torch.device('cuda:0')
        self.device=device


    
    def __getitem__(self,index):
        #Returns: i'th Cone Cropped Image  
        height, width, _ = self.image.shape
        
        x=self.x[index][1][0]
        y=self.x[index][1][1]
        w=self.x[index][1][2]
        h=self.x[index][1][3]

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_of_cone = self.image[y1:y2,x1:x2]
        resized_image_of_cone = cv2.resize(image_of_cone, (80,80))
        resized_image_of_cone = resized_image_of_cone.transpose((2, 0, 1)) / 255.0
        resized_image_of_cone = torch.from_numpy(resized_image_of_cone).type('torch.FloatTensor').to(device)
        return resized_image_of_cone , x, y, w, h

    def __len__(self):
        return self.n_samples


"""  
my_dataset=CustomDataset()

total_samples=len(my_dataset)
print(total_samples)
my_batch_size=total_samples

#now do dataloader
dataloader=DataLoader(dataset=my_dataset,batch_size=my_batch_size,shuffle=False,num_workers=2)
#num_workers=2 makes it faster
"""


"""

data_iterate=iter(dataloader)   #helps us iterate thru the dataloader
data_batch=data_iterate.next()   everytime we use this, the next batch of data is assigned to  data_batch
f,l=data_batch
print(f,l)

"""


"""
#to iterate thru the data and send for training/etsting/validation in batches

for i, (inputs,labels) in enumerate(dataloader):

   pass
"""   