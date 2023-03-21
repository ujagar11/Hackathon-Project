import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self,min_detectionCon = 0.5):
        self.min_detectionCon = min_detectionCon
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFace.FaceDetection(self.min_detectionCon)


    def findFaces(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        # print(self.results)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(detection.score)
                # print(id,detection)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                h,w,c = img.shape
                # mpDraw.draw_detection(img,detection)
                # to get the pixel values of bounding box
                bbox = int(bboxC.xmin * w),int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h) 
                bboxes.append([id,bbox,detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img,f'{int(detection.score[0] * 100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

        return bboxes,img
    
    def findLandmarks(self,img,draw=True):
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                landmarks = detection.location_data.relative_keypoints
                h,w,c = img.shape

                # left and right eye 
                rightEye = int(landmarks[0].x*w),int(landmarks[0].y*h)
                cv2.circle(img,rightEye,6,(255,0,0),cv2.FILLED)
                leftEye = int(landmarks[1].x*w),int(landmarks[1].y*h)
                cv2.circle(img,leftEye,6,(255,0,0),cv2.FILLED)
                cv2.line(img,leftEye,rightEye,(0,255,0),1)

                # left and right ear 
                rightEar = int(landmarks[4].x*w),int(landmarks[4].y*h)
                cv2.circle(img,rightEar,6,(255,0,0),cv2.FILLED)

                leftEar = int(landmarks[5].x*w),int(landmarks[5].y*h)
                cv2.circle(img,leftEar,6,(255,0,0),cv2.FILLED)
                cv2.line(img,leftEar,leftEye,(0,255,0),2)
                cv2.line(img,rightEar,rightEye,(0,255,0),2)

                # Nose
                nosetip = int(landmarks[2].x*w),int(landmarks[2].y*h)
                cv2.circle(img,nosetip,6,(255,0,0),cv2.FILLED)
                cv2.line(img,nosetip,rightEye,(0,255,0),2)
                cv2.line(img,nosetip,leftEye,(0,255,0),2)
                cv2.line(img,nosetip,rightEar,(0,255,0),2)
                cv2.line(img,nosetip,leftEar,(0,255,0),2)


                # Mouth
                mouth = int(landmarks[3].x*w), int(landmarks[3].y*h)
                cv2.circle(img,mouth,6,(255,0,0),cv2.FILLED)
                cv2.line(img,mouth,leftEye,(0,255,0),2)
                cv2.line(img,mouth,rightEye,(0,255,0),2)
                cv2.line(img,mouth,nosetip,(0,255,0),2)
        return img,landmarks        






    def fancyDraw(self,img,bbox,l = 30, thickness = 6,rt = 1):
        x,y,w,h = bbox
        # bottom right point
        x1,y1 = x+w,y+h

        cv2.rectangle(img,bbox,(255,0,255),rt)
        # Top left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),thickness)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),thickness)
        # Top right x1,y1
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),thickness)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),thickness)
        # Bottom left
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),thickness)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),thickness)
        # Bottom Right
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),thickness)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),thickness)
        return img

def main(): 
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(640,480))

        bboxes,img = detector.findFaces(img,draw=False)
        img,lm = detector.findLandmarks(img)
        # print(bboxes)
        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime

        cv2.putText(img,f'FPS : {int(fps)}',(40,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        cv2.imshow("Image1",img)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()