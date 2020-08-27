import numpy as np
import cv2

test = 'C:/Users/myip7/Desktop/test/dual_test.mp4'
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

def testDevice(source):
    cap = cv2.VideoCapture(source)#,cv2.CAP_MSMF)
    # print(cv2.CAP_PROP_FOURCC)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)

testDevice(0 + cv2.CAP_DSHOW) # no printout
# testDevice(1) # prints message if no source

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(ret)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# import time
# import cv2
#
# from tensorflow.keras import Input, Model
# from darknet import darknet_base
# from predict import predict, predict_with_yolo_head
#
#
# INCLUDE_YOLO_HEAD = True
#
# stream = cv2.VideoCapture(0)
#
# inputs = Input(shape=(None, None, 3))
# outputs, config = darknet_base(inputs, include_yolo_head=INCLUDE_YOLO_HEAD)
# model = Model(inputs, outputs)
#
#
# while True:
#     # Capture frame-by-frame
#     grabbed, frame = stream.read()
#     if not grabbed:
#         break
#
#     # Run detection
#     start = time.time()
#
#     if INCLUDE_YOLO_HEAD:
#         output_image = predict(model, frame, config)
#     else:
#         output_image = predict_with_yolo_head(model, frame, config, confidence=0.3, iou_threshold=0.4)
#
#     # output_image = frame
#     end = time.time()
#     print("Inference time: {:.2f}s".format(end - start))
#
#     # Display the resulting frame
#     cv2.imshow('', output_image)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# stream.release()
# cv2.destroyAllWindows()