import dlib
import cv2
import mysql.connector
import datetime

mydb = mysql.connector.connect(host="localhost", user="root", password="**", database="attendance")
mycursor = mydb.cursor()

Id = input('Enter ID: ')
name = input('Enter Name: ')

if(Id.isnumeric() and name.isalpha()):
    cam = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    for i in range(10):
        ret, img = cam.read()
        cv2.imwrite('TrainingImage/' + name.lower() + '.' + Id + '.' + str(i) + '.jpg', img)
        cv2.waitKey(100)
    cam.release()
    cv2.destroyAllWindows()
    sqlFormula = "INSERT INTO november (name) VALUES (%s)"
    data = (name,)
    mycursor.execute(sqlFormula,data)
    mydb.commit()
        
    
for i in range(10):
    gray = cv2.imread('TrainingImage/' + name.lower() + '.' + Id + '.' + str(i) + '.jpg')
    faces = detector(gray, 0)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite('TrainingImageGray/' + name.lower() + '.' + Id + '.' + str(i) + '.jpg', gray[y:y+h,x:x+w])
