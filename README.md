# FaceRecognition
Face Recognition using OpenCV with Harr-like feature. 

The data for the project is not large, First I decided to collect data in a naive way like go around and ask my coworker to give me some of their images that have their face . But I gradually relize that that's not an efficient way to collect data for face recognition and i dont have time to label all the face from a large picture. So I had to find a way to directly cut faces from a live video. In that way i can collect multi data at the same time and dont worry about spending on labling. But everything has pros and cons, because the data collected under the same condition (like the light, the type of hair,..) this makes the model more likely to be overfitting . I take about 60-100 pictures for each people. You can find this code via "FaceRecognition/take_image_and_train.ipynb"

 I used  haar-like to extract features of  faces for face detection and cv2 for recognition. This is not a state-of-the-art method and the accuracy just ranges from 60%.
 
Can not use camera on openCv for web deployment. I solved this problem  by using Js+canvas: Take images frame from camera  using Js and give it to canvas. Then using flask API to move it to the server, here we process the image and return the image result, which is an image that has  faces framed bounding box with name and acc was compressed to base64 image type.

The problem when using Js+canvas:the image is compressed and extracted many times. This makes the color of the final image not axactly the same as the original image.






