import cv2
import csv

Id = input('Enter ID: ')
name = input('Enter Name: ')

if(Id.isnumeric() and name.isalpha()):
    cam = cv2.VideoCapture(0)
    harcascadePath = r"C:\Users\User\haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    for i in range(61):
        ret, img = cam.read()
        cv2.imwrite('TrainingImage/' + name.lower() + '.' + Id + '.' + str(i) + '.jpg', img)
        cv2.waitKey(100)
    cam.release()
    cv2.destroyAllWindows()
    
for i in range(61):
    gray = cv2.imread('TrainingImage/' + name.lower() + '.' + Id + '.' + str(i) + '.jpg', 0)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #Saving the captured face in the dataset folder TrainingImage
        cv2.imwrite('TrainingImageGray/' + name.lower() + '.' + Id + '.' + str(i) + '.jpg', gray[y:y+h,x:x+w])
    row = [Id , name]
with open(r'C:\Users\User\Downloads\StudentDetails.csv','a+') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()


    




