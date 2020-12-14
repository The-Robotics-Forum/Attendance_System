import cv2
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/Trainner.yml")
cam = cv2.VideoCapture(0)
harcascadePath = "haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier(harcascadePath)

font=cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

df=pd.read_csv("StudentDetails/StudentDetails.csv")

while(True):
    ret1, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        name=df.loc[df['Id'] == Id]['Name'].values
        name_get=str(Id)+"-"+name
        cv2.putText(img,str(name_get),(x+w,y+h),font,0.5,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow('img',img)
    if (cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()