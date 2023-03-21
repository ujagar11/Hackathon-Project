# modules
import cv2
import mediapipe as mp
import time 
import FaceDetectionModule as fdm
import keys
from twilio.rest import Client
#capturing

cap = cv2.VideoCapture(0)
ig= cv2.imread('1.png')
#extracting coco dataset to list

className=[]
classfile= "coco.names"
with open (classfile,'rt') as f:
    className=f.read().rstrip('\n').split('\n')

#getting the ssd and weights

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt '
weightspath='frozen_inference_graph.pb'

#detecting uppers files

net=cv2.dnn_DetectionModel(weightspath,configPath)

#setting input measures


net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

# creating object of Face detector
detector = fdm.FaceDetector() 

#processing and detection
flag=0
while True:
      success,img= cap.read()
      h,w,c=img.shape
      bboxes,img= detector.findFaces(img,draw=False)
      img,lm = detector.findLandmarks(img)
      classIds,confs,bbox= net.detect(img,confThreshold=0.5)
    #   print(classIds,bbox)
    
      if len(classIds) != 0:
          for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
               
               cv2.rectangle(img,box,color=(0,255,0),thickness=0)
               if len(className) > classId-1:
                      # print(className[classId-1])
                      if className[classId-1] == "scissors":
                            cv2.putText(img, "Mobile", (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            flag=1
                      elif(className[classId-1]=="snowboard"):
                             continue
                      
                      else:
                          cv2.putText(img, className[classId-1].upper(), (box[0]+10, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)   
                          
                      if int(lm[4].x*w) >= int(lm[5].x*w)/2*1.65:
                          print("invalid")
                          flag==1        
                         
               else:
                   print("Class ID not found in className list.")

          if flag==1:
            #    client = Client(keys.account_sid,keys.auth_token)
            #    content = "Unnecessary Behaviour"
            #    message = client.messages.create(body=content,from_=keys.twilio_number,to=keys.target_number)
            #    print(message.sid)
              #  time.sleep(5)
             
               cv2.imshow("image",ig)
               cv2.waitKey(900)
               break
 
      cv2.imshow("Output",img)
      if cv2.waitKey(1) == 27:
           break

        
