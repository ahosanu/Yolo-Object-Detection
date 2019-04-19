import numpy as np
import imutils
import time
import cv2
import os
OutPutFile = "AhosanVideo.avi"
thresold = .2
LABELS = open('cfg/obj.names').read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("cfg/yolo-obj.cfg", "yolo-obj_last.weights")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

writer = None
(W, H) = (None, None)

def Obj(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] Complete took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > thresold:
                
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresold, thresold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 7), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    return image

cam = cv2.VideoCapture(0)
while True:
    if cv2.waitKey(1) == 27:
        writer.release()
        cv2.destroyAllWindows() 
        break  # esc to quit
    ret_val, img = cam.read()
    img = Obj(cv2.flip(img, 1))
    cv2.imshow('my webcam', img)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(OutPutFile, fourcc, 30, (img.shape[1], img.shape[0]), True)
    writer.write(img)
        
