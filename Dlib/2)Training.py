import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from imutils import paths
import pickle

path='TrainingImageGray'
imagePaths= list(paths.list_images(path))

knownEncodings=[]
knownNames=[]

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    name = imagePath.split(os.path.sep)[1]
    name = name[:-9]
    image= cv2.imread(imagePath)
    rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes= face_recognition.face_locations(rgb, model='cnn')
    encodings= face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("[INFO] serializing encodings...")
data={'encodings': knownEncodings, 'names': knownNames}
f= open('encodings.pickle', 'wb')
f.write(pickle.dumps(data))
f.close()