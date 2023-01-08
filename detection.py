#neffati ameni" 
#python detection.py --image images/car2.jpg --cfg cfg/yolov3-custom.cfg --weights backup/yolov3-custom_last.weights --classes data/tunisia_licence_plate.names"

import numpy as np
import os
import cv2 
import argparse
import time


start_time = time.perf_counter()

confThreshold = 0.7 
nmsThreshold = 0.6

ap = argparse.ArgumentParser(description='Object Detection using YOLOv3 in OPENCV')
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--cfg', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

# Load your class labels in which our YOLO model was trained on

labelsPath = (args.classes)
LABELS = open(labelsPath).read().strip().split("\n")

print('loading model, please wait')
# Loading the neural network framework Darknet (YOLO was created based on this framework)
net = cv2.dnn.readNetFromDarknet(args.cfg,args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print('model loaded')

# Get the names of the output layers
def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# Draw the predicted bounding box
def drawPred(frame,classes,classId, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv2.rectangle(frame, (left, top), (right, bottom), (26, 7, 248),2)

	label = '%.2f' % conf

	# Get the label for the class name and its confidence
	if classes:
		assert(classId < len(classes))
		label = '%s:%s' % (classes[classId], label)

	#Display the label at the top of the bounding box
	labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1)
	top = max(top, labelSize[1])
	#cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_DUPLEX, 1, (27,102,244),2)


# Remove the bounding boxes with low confidence using non-maximum suppression
def postprocess(frame, outs,confThreshold,nmsThreshold):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	classIds = []
	confidences = []
	boxes = []
	# Scan through all the bounding boxes output from the network and keep only the
	# ones with higher confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				print("accuracy:",confidence)
				print(LABELS)
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	detections=[]
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		right=left + width
		bottom=top + height
		result={
			"rect":(left, top,right,bottom),
			"classId":classIds[i],
			"confidence":confidences[i]
		}
		detections.append(result)
	return detections




    # Create the function which predict the frame input
def predict(image):
    # initialize a list of colors to represent each possible class label
    np.random.seed(50)
    COLORS = np.random.randint(125, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]
    
    # determine only the "ouput" layers name which we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    detections = net.forward(getOutputsNames(net))
    detections=postprocess(image, detections,0.0,0.5)


 
    crop_image=None
    for detection in detections:
      (startX, startY, endX, endY) =detection["rect"]
	  
      confidence=detection["confidence"]
      classId=detection["classId"]
      crop_image =image[startY:endY , startX:endX]
      drawPred(image,LABELS,classId, confidence, startX, startY, endX, endY)
      # display the prediction

    # show the output image
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    return (image,crop_image)


# Execute prediction on a single image
img = cv2.imread(args.image)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pred_img,crop_image=predict(img)
print("Done processing ")
print("car LP detected in : %s seconds ---" % (time.perf_counter() - start_time))
#display vehicule image with bounding box on plate
cv2.imshow('result',pred_img)
cv2.imwrite('result.jpg',pred_img)
#display the licence plate
cv2.imshow('cropped',crop_image)
cv2.imwrite('cropped.jpg',crop_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


 

