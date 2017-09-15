import cv2
import numpy as np
import glob
import pickle

from pdf2table.imageProcessing import imageEdit,detectLine,createSkeletonImage

def insert_template(path):
    try:
        with open('count', 'rb') as f:
            count = pickle.load(f)
    except FileNotFoundError:
        count=1
   # print(count)
    cv_img = []
    #print(path+"*.jpg")
    for img in glob.glob(path+"*.jpg"):
        n = cv2.imread(img)
        cv_img.append(n)
    for i in range(len(cv_img)):
        cv_img[i]=createSkeletonImage.create_skeleton(cv_img[i])
        cv_img[i]=cv2.cvtColor(cv_img[i],cv2.COLOR_BGR2GRAY)
    shape=[]
    height = 0 
    width = 0
    for i in range(len(cv_img)):
        shape.append(cv_img[i].shape[:2])
	#h2, w2 = image2.shape[:2]
        if shape[i][0] > height:
            height=shape[i][0]
        
        width+=shape[i][1]
    #print(height)
    #print(width)
    image_template=np.zeros((height,width), np.uint8)   
    temp=0
    for i in range(len(shape)):
        #print(type(shape[i][0]))
        image_template[:shape[i][0], temp:temp+shape[i][1]]=cv_img[i]
        temp=shape[i][1]

    #print(image_template.shape)
    brisk = cv2.BRISK_create(thresh= 10, octaves= 4)
    keyPoints= brisk.detect(image_template, None)
    keyPoints, descriptors = brisk.compute(image_template, keyPoints)
    print(descriptors.shape)
    write_in_file(descriptors,count)

def write_in_file(descriptors,count):
    with open("template "+str(count),'wb') as f:
        pickle.dump(descriptors,f)
    count+=1
    with open("count",'wb') as f:
        pickle.dump(count,f)

if __name__==__main__:
	insert_template("Paste path to your image folder")
