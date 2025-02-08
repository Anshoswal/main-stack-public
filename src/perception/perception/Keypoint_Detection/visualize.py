import os
import pandas as pd
import numpy as np
import cv2

def draw(src, points):

    if src[-3:] != 'jpg':
        src = src + '.jpg'
    path = os.path.join('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset',src)
    # print(path)

    img = cv2.imread(path)
    colors = [[255, 255, 255], [147, 20, 255], [255, 0, 0], [0, 0, 0], [0, 100, 0], [211,0,148], [0, 0, 255]]
    # [white, pink, blue, black, green, purple, red]

    if img is not None:
        res=img.copy()
        for idx,point in enumerate(points):
            coords = pd.eval(point)
            cv2.circle(res,coords,2,colors[idx],-1)

        res = cv2.resize(res, (200,400))
        cv2.imshow(src, res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

names = os.listdir('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset') #path to directory containg images
print(len(names))

# labels = pd.read_csv('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\rektnet_label.csv') #path to labels csv
# labels = pd.read_excel('output_test.xlsx')
labels = pd.read_excel('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\outputs\\october-2021-experiments\\geo_loss_train\\model_predictions.xlsx')

fnames = list(labels.values[:, 0])
points = labels.values[:, 2:9]
# print(points)
# print(points.shape[0])

test = fnames #[0:3] #select image indices to visualize
# print(test)

for fname in test:
    fname_idx = fnames.index(fname)
    fpoints = points[fname_idx]
    # print(fname)
    # print(fpoints)
    draw(fname, fpoints)