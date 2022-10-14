from PIL import Image
from flask import Flask, jsonify,render_template, request
from flask_jsglue import JSGlue
import base64
import io
import numpy as np

import cv2
import os

app=Flask(__name__)
JSGlue(app)

labels=[]

for labl in os.listdir("images"):
    labels.append(labl)

path='images'

recog=cv2.face.LBPHFaceRecognizer_create()
recog.read("training_index.yml")
facedetect=cv2.CascadeClassifier('haarcascade_frontalFace_default.xml')

def generate_frames(frame):
    recog=cv2.face.LBPHFaceRecognizer_create()
    recog.read("training_index.yml")
    facedetect=cv2.CascadeClassifier('haarcascade_frontalFace_default.xml')

    Gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    print("RGB : ",Gray_frame.shape)
    faces=facedetect.detectMultiScale(Gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    for x,y,w,h in faces:
            print("Finded Ur Face")
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            name_id,acc=recog.predict(Gray_frame[y:y+h,x:x+w])
            if name_id:
                cv2.putText(frame,labels[name_id] + str(round(acc,2)),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                print("ID :",labels[name_id])
            else:
                cv2.putText(frame,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1,cv2.LINE_AA)    
                print("Unknow")

    r,buffer=cv2.imencode('.jpg',frame)
    frame=buffer.tobytes()
    # yield (labels[name_id])
    # yield(b'--frame\r\n'
    #     b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/FaceRecognition',methods=["POST","GET"])
def FaceRecognition():
    print('==================================')
    if request.method=='POST':
        str_dta=request.data.decode("UTF-8")
        code_data=str_dta.split(',')[1]
        b64_data=base64.b64decode(code_data)
            
        image=Image.open(io.BytesIO(b64_data))
        image=image.convert('RGB')

        frame=np.array(image)
      
        Gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        print("Gray image : ",Gray_frame.shape)

        faces=facedetect.detectMultiScale(Gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for x,y,w,h in faces:
                print("Finded Ur Face")
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                name_id,acc=recog.predict(Gray_frame[y:y+h,x:x+w])
                if name_id:
                    cv2.putText(frame,labels[name_id] + str(round(acc,2)),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                    print("ID :",labels[name_id])
                else:
                    cv2.putText(frame,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1,cv2.LINE_AA)    
                    print("Unknow")

        r,buffer=cv2.imencode('.jpg',frame)
        image=buffer.tobytes()
        image=base64.b64encode(image).decode("UTF-8")
        print(len(image))

        return jsonify({'image':image})
    else:
        return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)


