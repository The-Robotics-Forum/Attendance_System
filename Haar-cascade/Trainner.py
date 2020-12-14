from PIL import Image
import numpy as np
import os
import cv2

def ImagesAndNames(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empty face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #Loading the images in Training images and converting it to gray scale
        g_image=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        image_ar=np.array(g_image,'uint8')
            #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(image_ar)
        Ids.append(Id)
    return faces,Ids

recognizer = cv2.face.createLBPHFaceRecognizer_create()
faces,Id = ImagesAndNames("TrainingImageGray")
recognizer.train(faces, np.array(Id))
recognizer.save("recognizers/Trainner.yml")