import face_recognition
import pickle
import cv2
from datetime import datetime

print("[INFO] loading encodings...")
data= pickle.loads(open('encodings.pickle', 'rb').read())

def mark_attendance(n):
    with open('StudentDetails.csv', 'r+') as f:
        myDataList= f.readlines()
        print(myDataList)
        nameList=[]
        for line in myDataList:
            name= line.split(',')[0]
            nameList.append(name)

        if n not in nameList:
            now= datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{n},{dtString}')
            
            date1 = "NOV_"+datetime.date.today().strftime("%d-%m-%Y")[:2]
            time1 = str(datetime.datetime.now().strftime("%H:%M:%S"))
            query = "UPDATE  november SET  "+date1+" = '"+time1+"' WHERE name = '"+name+"'"
            insert = (date1,time1,name)
            mycursor.execute(query)
            mydb.commit()

cap= cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # img = captureScreen()
    image = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
                                            model='cnn')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = 'Unknown'

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        mark_attendance(name)

    cv2.imshow('img',image)
    if (cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()