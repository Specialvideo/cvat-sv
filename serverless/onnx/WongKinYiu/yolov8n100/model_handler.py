# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import base64
from ultralytics import YOLO
from skimage.measure import approximate_polygon, find_contours

class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model='yolon_100.pt')
        self.labels = labels
        self.h=0
        self.w=0

    def load_network(self, model):
        self.model = YOLO(model)
        
    
    def infer(self, image, threshold):
        results = self.model([image], stream=True, conf = 0.25, iou = 0.5, retina_masks=True)

        yolo_classes = [
            "Anchovy", "Olives", "Salame", "Red_Pepper", 
            "Yellow_Pepper"
        ]
        
        objects = []
        for row in results:
            for mask, box in zip(row.masks,row.boxes):
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 =x1.item(),y1.item(),x2.item(),y2.item()
                if len(mask.xy[0])>2:
                    objects.append([0,0,0,0,yolo_classes[int(box.cls[0])],0.1,None,mask.xy[0]])

        
        #objects.sort(key=lambda x: x[5], reverse=True)
        #nms = []
        #while len(objects)>0:
        #    nms.append(objects[0])
        #    objects = [object for object in objects if self.iou(object,objects[0])<0.8]

        results = []
        for elem in objects:
            results.append({
                        "confidence": str(elem[5]),
                        "label": elem[4],
                        "points":[int(item) for row in elem[7] for item in row],
                        #"blob": Image.fromarray(np.array(elem[6])),
                        "type": "mask",
                    })
        
        return results
