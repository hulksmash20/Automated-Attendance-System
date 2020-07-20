import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime


path='E:\Project\Face detection\Images'
images=[]
classnames=[]
mylist=os.listdir(path)
#print(mylist)
for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
#print(images)
#print(classnames)

def encoding(images):
    encodedlist=[]
    for i in images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=fr.face_encodings(i)[0]
        encodedlist.append(encode)
    return encodedlist

encodelistknown= encoding(images)
#print('done encoding')

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines()
        namelist=[]
        for line in myDataList:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtstring= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
            
            
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame= fr.face_locations(imgS)
    encodeCurFrame= fr.face_encodings(imgS,facesCurFrame)
    
    
    for encodeface,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches= fr.compare_faces(encodelistknown,encodeface)
        faceDis= fr.face_distance(encodelistknown,encodeface)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name=classnames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1= faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)