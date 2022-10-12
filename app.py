from flask import Flask,render_template,Response

import cv2
import os

app=Flask(__name__)
labels=[]

for labl in os.listdir("images"):
    labels.append(labl)

path='images'

def generate_frames():
    recog=cv2.face.LBPHFaceRecognizer_create()
    recog.read("training_index.yml")

    video=cv2.VideoCapture(1)
    facedetect=cv2.CascadeClassifier('haarcascade_frontalFace_default.xml')
    

    while True:
        ret,frame=video.read()
        RGB_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(RGB_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        for x,y,w,h in faces:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            name_id,acc=recog.predict(RGB_frame[y:y+h,x:x+w])

            if name_id:
                cv2.putText(frame,labels[name_id] + str(round(acc,2)),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
            else:
                cv2.putText(frame,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1,cv2.LINE_AA)    
        
        r,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
