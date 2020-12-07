#!/usr/bin/env python
# coding: utf-8

# In[11]:

from __future__ import division, print_function, absolute_import
import face_recognition
import cv2
import numpy as np
#import tensorflow as tf
import time

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)


def main():
    video_capture = cv2.VideoCapture(r"small_camera_test_video.mp4")

    print(video_capture)

    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    # Initialize some variables
    face_locations = []
    face_encodings = []
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter('face_recognition_test.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))
    start_time = time.time()
    frame_num = 0

    try:
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            print(ret)
            if ret:
                frame_num += 1
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                if frame_num % 2 == 0:
                    print(" I am here")
                    rgb_small_frame = frame[:, :, ::-1]
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
                    print(face_locations)
                    #print("face_locations {}".format(face_locations))
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    #print("face_encodings {}".format(len(face_encodings)))
                    face_names = []
                    for face_encoding in face_encodings:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        #print("face_distance {}".format(face_distances))
                        if len(face_distances) > 0 and face_distances[np.argmin(face_distances)] < .6:
                            name = known_face_names[np.argmin(face_distances)]
                        else:
                            known_face_encodings.append(face_encoding)
                            name = 'face'+str(len(known_face_encodings)-1)
                            print(name)
                            known_face_names.append(name)
                        face_names.append(name)


                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, name, (left + 6, bottom - 6),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                    cv2.imshow("img", frame)

            # Display the resulting image
                out.write(frame)
            else:
                break


        # Release handle to the webcam
        print("Time Taken to complete the detection {}".format(time.time()-start_time))
        video_capture.release()
    except KeyboardInterrupt:
            out.release()
            video_capture.release()


if __name__ == '__main__':
    main()





# %%
