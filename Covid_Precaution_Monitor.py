# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 00:26:43 2020

@author: Akash
"""
#importing required libraries
import os
import numpy as np
import cv2
import face_detection
from sklearn.cluster import DBSCAN
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import tqdm

#path to the input video
FILE_PATH = "videos/sample_videos.mp4"

#initializing a face detector
detector = face_detection.build_detector("DSFDDetector", confidence_threshold = 0.5, nms_iou_threshold = 0.3)

#loading pretrained face mask classifier(keras model)
mask_classifier = load_model("models/ResNet50_Classifier.h5")

#setting the safe distance in pixel units(may vary according to each video)
threshold_distance = 150

#---Analyzing the Video---#
#loading YOLOv3
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")

#loading COCO classes
classes = []
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#fetching video properties
cap = cv2.VideoCapture(FILE_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

#creating directory for storing results
os.mkdir("results")
os.mkdir("results/extracted_faces")
os.mkdir("results/extracted_persons")
os.mkdir("results/frames")

#initializing output video stream
out_stream = cv2.VideoWriter(
    "result.mp4", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (int(width), int(height)))

print("Processing Frames :")
for frame in tqdm.tqdm(range(int(n_frames))):
    
    #capturing frame-by-frame
    ret, img = cap.read()
    
    #checking for EOF
    if ret == False:
        break;
    
    #getting frame dimensions
    height, width, channels = img.shape
    
    #detecting objects in the frame with YOLOv3
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    #storing detected objects with labels, bounding_boxes and their confidences
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #getting center, height, width of the box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                #topleft co-ordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    #initializing empty lists for storing bounding boxes of people and their faces
    persons = []
    masked_faces = []
    unmasked_faces = []
    
    #working on detected persons in the frame
    for i in range(len(boxes)):
        if i in indexes:
            
            box = np.array(boxes[i])
            box = np.where(box < 0, 0, box)
            (x, y, w, h) = box
            
            label = str(classes[class_ids[i]])
            
            if label == 'person':
                persons.append([x, y, w, h])
                
                #commenting to improve processing speed(uncomment if required)
                #saving image of cropped person
                #cv2.imwrite("results/extracted_persons" + str(frame) + "_" + str(len(persons)) + ".jpg", img[y : y + h, x : x + w])
                
                #detecting face in the person
                person_rgb = img[y : y + h, x: x + w,::-1] #crop & BGR to RGB
                detections = detector.detect(person_rgb)
                
                #if a face is detected
                if detections.shape[0] > 0:
                    detection = np.array(detections[0])
                    detection = np.where(detection < 0, 0, detection)
                    
                    #calculating co-ordinates of the detected face
                    x1 = x + int(detection[0])
                    x2 = x + int(detection[2])
                    y1 = y + int(detection[1])
                    y2 = y + int(detection[3])
                    
                    try :
                        #crop & BGR to RGB
                        face_rgb = img[y1 : y2, x1 : x2, : : -1]
                        
                        #preprocessing the image
                        face_arr = cv2.resize(face_rgb, (224, 224), interpolation = cv2.INTER_NEAREST)
                        face_arr = np.expand_dims(face_arr, axis = 0)
                        face_arr = preprocess_input(face_arr)
                        
                        #predicting if the face is masked or not
                        score = mask_classifier.predict(face_arr)
                        
                        #determining and storing the results
                        if score[0][0] < 0.5:
                            masked_faces.append([x1, y1, x2, y2])
                        else:
                            unmasked_faces.append([x1, y1, x2, y2])
                        
                        #commenting to improve processing speed(uncomment if required)    
                        #saving image of the cropped face
                        #cv2.imwrite("results/extracted_faces" + str(frame) + "_" + str(len(persons)) + ".jpg", img[y1 : y2, x1: x2])
                        
                    except:
                        continue
    
    #calculating coordinates of people detected and find clusters using DBSCAN
    person_coordinates = []
    
    for p in range(len(persons)):
        person_coordinates.append((persons[p][0] + int(persons[p][2] / 2), persons[p][1] + int(persons[p][3] / 2)))
        
    clustering = DBSCAN(eps = threshold_distance, min_samples = 2).fit(person_coordinates)
    isSafe = clustering.labels_
        
    #count
    person_count = len(persons)
    masked_face_count = len(masked_faces)
    unmasked_face_count = len(unmasked_faces)
    safe_count = np.sum((isSafe == -1)*1)
    unsafe_count = person_count - safe_count
        
    #showing clusters using red lines
    arg_sorted = np.argsort(isSafe)
        

    for i in range(1, person_count):
        if isSafe[arg_sorted[i]] != -1 and isSafe[arg_sorted[i]] == isSafe[arg_sorted[i-1]]:
            cv2.line(img, person_coordinates[arg_sorted[i]], person_coordinates[arg_sorted[i-1]], (0, 0, 255), 2)
        
    #placing bounding boxes on people in the frame
    for p in range(person_count):
        a, b, c, d = persons[p]
        
        #green if safe, red if unsafe
        if isSafe[p] == -1:
            cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (a, b), (a + c, b + d), (0, 0, 255), 2)
                
    #placing bounding boxes on faces
    for f in range(masked_face_count):
        a, b, c, d = masked_faces[f]
        #green because safe
        cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)
            
    for f in range(unmasked_face_count):
        a, b, c, d = unmasked_faces[f]
        #red because unsafe
        cv2.rectangle(img, (a, b), (c, d), (0, 0, 255), 2)
            
    #displaying the monitoring status in a black box at the top
    cv2.rectangle(img, (0, 0), (width, 50), (0, 0, 0), -1)
    cv2.rectangle(img, (1, 1), (width - 1, 50), (255, 255, 255), 2)
    
    xpos = 15
    
    string = "Total People = " + str(person_count)
    cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    xpos += cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
    
    string = " ( " + str(safe_count) + " Safe "
    cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    xpos += cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
    
    string = str(unsafe_count) + " Unsafe )"
    cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    xpos += cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
    
    string = " ( " + str(masked_face_count) + " Masked " + str(unmasked_face_count) + " Unmasked " + str(person_count - masked_face_count - unmasked_face_count) + " Unknown )"
    cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    #writing frame to the output file
    out_stream.write(img)
    
    #commenting to improve processing speed(uncomment if required)
    #saving the frame in frame_no.jpg format
    #cv2.imwrite("results/frames/" + str(frame) + ".jpg", img)
        
    #enabling exit on pressing Q key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

#releasing streams
out_stream.release()
cap.release()
cv2.destroyAllWindows()

#printing confirmation message        
print("DONE!")