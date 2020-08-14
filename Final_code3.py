# to run this program
#python Final_code3.py --shape-predictor shape_predictor_68_face_landmarks.dat \
 #   --encodings encodings.pickle \
  #  --output output/webcam_face_recognition_output3.avi \
   # --display 1

#python Final_code3.py --shape-predictor shape_predictor_68_face_landmarks.dat \
 #    --encodings encodings.pickle \
  #   --display 1

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import  face_utils
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from matplotlib import style
style.use ('fivethirtyeight')
import face_recognition
import numpy as np
import datetime
import argparse
import sqlite3 
import imutils
import random 
import pickle
import time
import dlib 
import cv2 

conn = sqlite3.connect('personinfo.db')
c = conn.cursor()

def create_table():
    c.execute ('CREATE TABLE IF NOT EXISTS personalinfo ( unix REAL, datestamps TEXT,\
        keyword TEXT, value REAL)')

def date_entry():
    c.execute("INSERT INTO personalinfo VALUES(151661161, '2019-05-14','python',8)")
    conn.commit()

def getsysdate():
    unix = time.time()
    sysdate = str(datetime.datetime.fromtimestamp(unix).strftime('%d-%m-%Y  %H:%M:%S'))
    return sysdate

def getinfo(Name):
    print("this is fun INFO value : " + Name)
    c.execute('SELECT * FROM People WHERE name ="' + str(Name) + '";')
    profile = None
    for row in c.fetchall():
        profile = row
        print(profile)
    return profile

def getID(Name):
    print("this is fun ID Val : " + Name)
    c.execute('SELECT ID FROM People WHERE name ="' + str(Name) + '";')
    ID = None
    for row in c.fetchall():
        ID = row
        print(ID[0])
        print("THIS IS STRING ID = " + str(ID))
    return ID[0]

# def isDataExists(onlydate):
#     c.execute('SELECT IDs FROM Attendance WHERE Date ="' + str(onlydate) + '";')
#     atID = None
#     for row in c.fetchall():
#         atID = row
#         print(atID)
#     return atID

def isDataExists(onlydate,ID):
    c.execute('SELECT ID FROM People as P Attendance as A WHERE P."' + str(ID) + '" = A.IDs, AND A.Date ="' + str(onlydate) + '";')
    atID = None
    for row in c.fetchall():
        atID = row
        print(atID)
    return atID[0]

def doPresent(onlydate,onlytime,ID):
    pdate = str(onlydate)
    ptime = str(onlytime)
    pID = int(ID)
    print("this is fun PRESENT : " + str(pID),pdate,ptime)
    c.execute("INSERT INTO Attendance(IDs,Date,time,Status) VALUES(?,?,?,'P')",
        (pID,pdate,ptime))
    conn.commit()
    print("--------------------------------------------------------------DONE ")

def doCount(IDs):
    c.execute('SELECT count(*) FROM Attendance WHERE IDs ="' + str(IDs) + '";')
    count = None
    for row in c.fetchall():
        count = row
        print(count[0])
    return count[0]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-e", "--encodings",required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output",type=str,
    help="path to output video")
ap.add_argument("-y", "--display",type=int,default=1,
    help="whether or not to diaplay output fame tp screen")
ap.add_argument("-d", "--detection-method",type=str,default="hog",
    help="face detection model to use : either 'hog' or 'cnn' ")
args = vars(ap.parse_args())

################################################# DECLARATIONS ######################################################
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAME = 2

COUNTER = 0
TOTAL = 0
FLAG = 0

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream...")
vs= VideoStream(src=0).start()
writer = None
time.sleep(2.0)

#############################################  DETECT THE PERSON FIRST #############################################
try:
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        r =frame.shape[1] / float(rgb.shape[1])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes = face_recognition.face_locations(rgb,
            model = args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        
        for encoding in encodings:

            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)

        for ((top, right, bottom, left),name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2, lineType = cv2.LINE_8)
            y = top - 15 if top -15 > 15 else top + 25
            x = bottom + 15 if bottom + 15 < 15 else bottom - 15


############################################# DETECT THE EYE_BLINK COUNTS ###########################################
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHaull = cv2.convexHull(leftEye)
            rightEyeHaull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHaull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHaull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

            else:
                if COUNTER >= EYE_AR_CONSEC_FRAME:
                    TOTAL += 1

                COUNTER = 0
################################# TRIAL #########################----------------------------------------
            profile = getinfo(name)
            
            if (profile != None):
                # for Y axis
                
                # cv2.putText(frame, "Name : "+ str(profile[0]), (left+1, y), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.75, (0, 255, 0), 1, lineType = cv2.LINE_8)
                # cv2.putText(frame, "Age : " + str(profile[1]), (left+1, y+30), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.75, (0, 255, 0), 1, lineType = cv2.LINE_8)
                # cv2.putText(frame, "Filed : " + str(profile[2]), (left+1, y+60), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.75, (0, 255, 0), 1, lineType = cv2.LINE_8)
                # cv2.putText(frame, "Clg : " + str(profile[3]), (left+1, y+90), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.75, (0, 255, 0), 1, lineType = cv2.LINE_8)
                
                # for X axis

                cv2.putText(frame, "Name : "+ str(profile[1]), (left+1, x+30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 255), 2, lineType = cv2.LINE_8)
                cv2.putText(frame, "Filed : " + str(profile[2]), (left+1, x+60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 0), 2, lineType = cv2.LINE_8)
                cv2.putText(frame, "Age : " + str(profile[3]), (left+1, x+90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 255), 2, lineType = cv2.LINE_8)
                cv2.putText(frame, "Clg : " + str(profile[4]), (left+1, x+120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, lineType = cv2.LINE_8)
                
                # rects = detector(gray, 0)

                # for rect in rects:
                #     shape = predictor(gray, rect)
                #     shape = face_utils.shape_to_np(shape)
                    
                #     leftEye = shape[lStart:lEnd]
                #     rightEye = shape[rStart:rEnd]
                #     leftEAR = eye_aspect_ratio(leftEye)
                #     rightEAR = eye_aspect_ratio(rightEye)

                #     ear = (leftEAR + rightEAR) / 2.0

                #     leftEyeHaull = cv2.convexHull(leftEye)
                #     rightEyeHaull = cv2.convexHull(rightEye)
                #     cv2.drawContours(frame, [leftEyeHaull], -1, (0, 255, 0), 1)
                #     cv2.drawContours(frame, [rightEyeHaull], -1, (0, 255, 0), 1)

                #     if ear < EYE_AR_THRESH:
                #         COUNTER += 1

                #     else:
                #         if COUNTER >= EYE_AR_CONSEC_FRAME:
                #             TOTAL += 1

                #         COUNTER = 0

                # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10,30),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                # cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # if(TOTAL > 2):
                #     ID = getID(name)
                #     print("ID of ----------------DETECTED_PERSON : " + str(ID))

                #     sysdate = getsysdate()
                #     onlydate = sysdate.split()
                #     print("TODAYS ------------------------DATE = " + onlydate[0],onlydate[1])

                #     atID = isDataExists(onlydate[0],ID)

                #     if(atID == None):
                #         print("[INFO] now we are doing your Attendance...")
                #         doPresent(onlydate[0],onlydate[1],ID)

                #     else:
                #         print("[INFO] YourAttendance for today is DONE..")
                #         cv2.putText(frame,"STAUS = Done", (300,60), cv2.FONT_HERSHEY_SIMPLEX,
                #              0.75, (0, 255, 255), 2, lineType = cv2.LINE_8)


                # ID = getID(name)
                # print("ID of Person : " + str(ID))
                # sysdate = getsysdate()
                # onlydate = sysdate.split()
                # print("TODAYS DATE = " + onlydate[0],onlydate[1])
                # atID = isDataExists(onlydate[0])
                # print(atID)
                # for i in range(len(atID)):
                #     if(3 == i ):
                #         print('Maro id = '+ID) 
                #         print(i)
                #         print('Madigayooooooo ID = '+ ID)
                #     else:
                #         print('No ID in LOL')

                
                # atcount = doCount(ID)

                # cv2.putText(frame,"ToT_AT: " + str(atcount), (320,30), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.75, (0, 255, 255), 2, lineType = cv2.LINE_8)

                # if((TOTAL > 2) and (atID == None)):
                #     print("[INFO] now we are doing your Attendance...")
                #     doPresent(onlydate[0],onlydate[1],ID)
                #     TOTAL = 0
                # else:
                #     cv2.putText(frame, "Your AT DONE Mr/Mrs "+ str(profile[1]), (120,120), cv2.FONT_HERSHEY_SIMPLEX,
                #         0.75, (0, 0, 255), 2, lineType = cv2.LINE_8)
        
#----------------------------------------------------------------------------------------
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
################################################### WRITE IN THE FOLDER ##############################################################
        if writer is None and args["output"] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 10,
                (frame.shape[1], frame.shape[0]), True)
            
        if writer is not None:
            writer.write(frame)
            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:   

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    vs.stop()
    c.close()
    conn.close()
