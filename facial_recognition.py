#!/usr/bin/env python3

#    Copyright 2021 Abdelfattah TOULAOUI
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Foobar is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import os
import numpy as np
from PIL import Image
import numpy as np
import tensorflow.lite as tflite

cascPathface = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)

clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

# Load tflite model
vectorizer = tflite.Interpreter(model_path='vectorizer.tflite')
vectorizer.allocate_tensors()

# A function to get the facial embedding from image
def get_face_embedding(face):
    face = face.astype(np.float32)
    face1 = np.expand_dims(face, 0)
    input_details = vectorizer.get_input_details()
    output_details = vectorizer.get_output_details()
    vectorizer.set_tensor(input_details[0]['index'], face1)
    vectorizer.invoke()
    return vectorizer.get_tensor(output_details[0]['index'])

# Crop the image and return it
def crop_around_face(image, rec):
    x, y, w, h = rec
    nh = h
    nw = w
    nx = x - (nw - w)//2
    ny = y - (nh - h)//2
    face = image[max(0, ny):ny+nh, max(0, nx):nx+nw]
    rows,cols,_ = face.shape
    ROI = cv2.copyMakeBorder(face,
            max(-ny, 0),
            nh-(max(-ny,0)+rows),
            max(-nx, 0),
            nw-(max(-nx,0)+cols),
            cv2.BORDER_CONSTANT,
            0)
    ROI = cv2.resize(ROI, (128, 128))
    ROI_grey = cv2.cvtColor(ROI,
            cv2.COLOR_BGR2GRAY)
    ROI_grey = cv2.equalizeHist(ROI_grey)
    return ROI_grey.reshape((128,128,1))

# Get facial embedding from image and rectangle
def from_face(image, face):
    face = crop_around_face(image, face)
    cv2.imshow('face', face)
    return get_face_embedding(face)

# Use OpenCV HAAR Cascade to find faces in an image
def find_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
            scaleFactor=1.75,
            minNeighbors=6,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

# Compare two facial embeddings
def compare_facial_embedding(face1, face2):
    diff = face1 - face2
    return np.square(diff).sum()

known_people = {}

# Get the embeddings from known_people
for f in os.listdir('known_people/'):
    try:
        file_path = 'known_people/%s'%f
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            face = find_faces(image)[0]
            known_people[f] = from_face(image, face)
    except:
        pass

people_list = list(known_people.keys())

print('Loaded %i people'%len(known_people))

video_capture = cv2.VideoCapture(0)
c = 1
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = find_faces(frame)
    # Loop over faces
    for face in faces:
        # Find the facial embedding
        embedding = from_face(frame, face)
        # Get similarity to known people
        similarity = [compare_facial_embedding(known_people[person],
            embedding) for person in people_list]
        # Get the minimum
        sim = np.argmin(similarity)

        # The threshold is .5
        # If over the threshold, make the box red
        # Else, make it green and print the name
        if similarity[sim] < .5:
            person = people_list[sim]
            color = (0, 255, 0)
        else:
            person = 'Unknown'
            color = (0, 0, 255)
        (x,y,w,h) = face

        # Draw the box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
        cv2.putText(frame, '%s (%f)'%(person, similarity[sim]),
                (x, y - 12), 0, 1e-3 * frame.shape[0], color, 3)
    # Show the image
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
