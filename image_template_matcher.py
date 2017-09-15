import cv2
import numpy as np
import glob
import pickle

from pdf2table.imageProcessing import imageEdit,detectLine,createSkeletonImage

def test_image(path):
    cv_img = []
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
        if shape[i][0] > height:
            height=shape[i][0]
        width+=shape[i][1]
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
    check_templates(descriptors)

def check_templates(descriptors):
    try:
        with open("count", 'rb') as f:
            #print('yes')
            count = pickle.load(f)
            #print(count)
            match_image(descriptors,int(count))
            #print('wwwww')
    except FileNotFoundError:
        print ('No Templates present to match the input image!!')
        print ('Descriptors of this image is being saved!!')
        count = 1
        with open("template "+str(count),'wb') as f:
            pickle.dump(descriptors,f)
        count+=1
        with open("count",'wb') as f:
            pickle.dump(count,f)


def match_image(descriptors,count):
    max_matches=0
    for i in range(1,count):
        with open("template "+str(i),'rb') as f:
            template_descriptors = pickle.load(f)
        print(template_descriptors.shape)
        FLANN_INDEX_LSH= 0
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        descriptors = np.array(descriptors.astype('float32'))
        template_descriptors = np.array(template_descriptors.astype('float32'))
        matches=flann.knnMatch(descriptors,template_descriptors,k=2)
        good = 0
        for j,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good=good+1
        if(good>max_matches):
            max_matches=good
            temp_num=i
            
    if(max_matches>1600):
        print("The match is found with template "+str(temp_num))
    else:
        print ('No match found !!')
        print ('Descriptors of this image is being saved!!')
        with open("template "+str(count),'wb') as f:
                pickle.dump(descriptors,f)
        count+=1
        with open("count",'wb') as f:
            pickle.dump(count,f)


if __name__==__main__:
	test_image("Paste path to your image folder")
