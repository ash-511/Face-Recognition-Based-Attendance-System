import tkinter as tk
from tkinter import *
from tkinter import Message ,Text
from playsound import playsound
import cv2,os
import shutil
import csv
import calendar
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import csv

window = tk.Tk()
window.title("Attendance MS")
playsound('Welcome-to-face-recognition-system.mp3')
path = 'blue.png'
img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")

message = tk.Label(window, text="Dashboard" ,bg="#1d2126"  ,fg="white"  ,width=22  ,height=36, font=('times', 10, 'italic bold')) 
message.place(x=0, y=0)
lbl = tk.Label(window, text="Facial Recognition",width=22  ,height=1  ,fg="white"  ,bg="#1d2126" ,font=('times', 10, ' bold ')) 
lbl.place(x=0, y=15)
lbl = tk.Label(window, text="Based",width=22  ,height=1  ,fg="white"  ,bg="#1d2126" ,font=('times', 10, ' bold ')) 
lbl.place(x=0, y=45)
lbl = tk.Label(window, text="Attendance Tracker",width=22  ,height=1  ,fg="white"  ,bg="#1d2126" ,font=('times', 10, ' bold ')) 
lbl.place(x=0, y=75)
lbl = tk.Label(window, text="______________________",width=22  ,height=1  ,fg="#1477b5"  ,bg="#1d2126" ,font=('times', 10, ' bold ')) 
lbl.place(x=0, y=295)

message = tk.Label(window, text=" " ,bg="#1a7eb0"  ,fg="white"  ,width=72  ,height=24,font=('times', 10, 'italic bold')) 
message.place(x=175, y=17)

message = tk.Label(window, text="New Users" ,bg="#1a7eb0"  ,fg="white"  ,width=25  ,height=2,font=('times', 20, 'italic bold')) 
message.place(x=200, y=130)

lbl = tk.Label(window, text="Enter ID:",width=13  ,height=1  ,fg="white"  ,bg="#1a7eb0" ,font=('times', 13, ' bold ')) 
lbl.place(x=215, y=195)

txt = tk.Entry(window,width=20  ,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=378, y=195)

lbl2 = tk.Label(window, text="Enter Name:",width=13  ,fg="white"  ,bg="#1a7eb0"   ,height=2 ,font=('times', 13, ' bold ')) 
lbl2.place(x=215, y=226)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=378, y=235)

message = tk.Label(window, text=" " ,bg="#1a7eb0"  ,fg="#1a7eb0"  ,width=72  ,height=7,font=('times', 10, 'italic bold')) 
message.place(x=175, y=425)

lbl3 = tk.Label(window, text="Attendance : ",width=16  ,fg="white"  ,bg="#1a7eb0"  ,height=2 ,font=('times', 13, ' bold ')) 
lbl3.place(x=175, y=425)

message2 = tk.Label(window, text="" ,fg="white"   ,bg="#1a7eb0",activeforeground = "green",width=28  ,height=3  ,font=('times', 13, ' bold ')) 
message2.place(x=383, y=455)
 

def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def TakeImages(): 
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('Facial Recognition',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        playsound('Dataset-Created.mp3')
        cam.release()
        cv2.destroyAllWindows() 
        res = "Dataset Created" 
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Dataset Trained Successfully"
    message.configure(text= res)
    playsound('Dataset-Trained-Successfully.mp3')

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)  
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])    
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
               
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('Facial Recognition',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    #fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    fileName="Attendance\Attendance_"+date+".csv"
    f= open(fileName,'w')
    writer=csv.writer(f)
    writer.writerow(attendance)
    f.close()

    with open(fileName,'a+',newline='') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(pd.DataFrame(columns=[Id,aa,date,timeStamp]))
         #   if row[0]==Id:
          #      break
           # writer.writerow(pd.DataFrame(columns=[Id,aa,date,timeStamp]))
        #for row in attendance:
            #writer.writerow(row)
    #attendance.to_csv(fileName,index=False)
    res="Attendance Updated"
    message.configure(text= res)
    playsound('Thank-you-Your-attendance-updated.mp3')
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)

def getReport():
    path = r'C:\Users\Saurabh\Documents\Shreya\Project1\Attendance'
    files = os.listdir(path)
    xls_files = []
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days = 7)
    for f in files:
        if f[-3:] == "csv":
            date = datetime.datetime.strptime(f[11:21], '%Y-%m-%d')
            if(date.date() > week_ago and date.date() <= today):
                xls_files.append(f)
    l = []
    l1 = []
    path1 = r'C:\Users\Saurabh\Documents\Shreya\Project1\StudentDetails\StudentDetails.csv'
    details = pd.read_csv(path1)
    l = details['Id']
    d = { i : 0 for i in l }
    for f in xls_files:
        df = pd.read_csv(path + '\\' + f)
        l1 = df['Id']
        for i in l:
            for j in l1:
                if i==j:
                    d[i] += 1
    l1 = d.values()
    perc = map(lambda x: (x/5)*100, l1)
    absent = map(lambda x: 5-x, l1)
    ts = time.time()  
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    res = list(zip(l, details['Name'], l1, absent, perc))            
    with open('Report\Report_'+date+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Name", "No. of days present","No. of days absent", "Percentage Attendance"])
        writer.writerows(res)
    file.close()
    res = "Weekly report generated"
    message.configure(text= res)
    playsound('Weekly-report-generated.mp3')

clearButton = tk.Button(window, text="-", command=clear  ,fg="white"  ,bg="midnight blue"  ,width=3  ,height=1 ,activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton.place(x=595, y=195)
clearButton2 = tk.Button(window, text="-", command=clear2  ,fg="white"  ,bg="midnight blue"  ,width=3  ,height=1, activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton2.place(x=595, y=235)    
takeImg = tk.Button(window, text="Register", command=TakeImages  ,fg="white"  ,bg="#1a7eb0"  ,width=29  ,height=5, activebackground = "aqua" ,font=('times', 12, ' bold '))
takeImg.place(x=705, y=153)
trainImg = tk.Button(window, text="Add Student to Database", command=TrainImages  ,fg="white"  ,bg="#1a7eb0"  ,width=29  ,height=5, activebackground = "gold" ,font=('times', 12, ' bold '))
trainImg.place(x=705, y=289)
trainImg = tk.Button(window, text="Generate Weekly Report", command=getReport  ,fg="white"  ,bg="#1a7eb0"  ,width=29  ,height=5, activebackground = "gold" ,font=('times', 12, ' bold '))
trainImg.place(x=705, y=425)
trackImg = tk.Button(window, text="Mark Attendance", command=TrackImages  ,fg="white"  ,bg="#1a7eb0"  ,width=29  ,height=5, activebackground = "lime" ,font=('times', 12, ' bold '))
trackImg.place(x=705, y=17)
window.mainloop()
