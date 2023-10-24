# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import base64
from skimage.measure import approximate_polygon, find_contours

class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="best.onnx")
        self.labels = labels
        self.h=0
        self.w=0

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    def intersection(self,box1,box2):
        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
        x1 = max(box1_x1,box2_x1)
        y1 = max(box1_y1,box2_y1)
        x2 = min(box1_x2,box2_x2)
        y2 = min(box1_y2,box2_y2)
        return (x2-x1)*(y2-y1)

    def union(self,box1,box2):
        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
        box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
        box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
        return box1_area + box2_area - self.intersection(box1,box2)

    def iou(self,box1,box2):
        return self.intersection(box1,box2)/self.union(box1,box2)

    def get_mask(self,row,box):
        mask = row.reshape(160,160)
        mask = self.sigmoid(mask)
        mask = (mask > 0.5).astype('uint8')*255
        x1,y1,x2,y2 = box
        mask_x1 = round(x1/self.w*160)
        mask_y1 = round(y1/self.h*160)
        mask_x2 = round(x2/self.w*160)
        mask_y2 = round(y2/self.h*160)
        mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
        img_mask = Image.fromarray(mask,"L")
        img_mask = img_mask.resize((int(round(x2-x1)),int(round(y2-y1))))
        mask = np.array(img_mask)
        return mask

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def to_cvat_mask(self,box, mask):
        xtl, ytl, xbr, ybr = box
        xtl = int(xtl)
        ytl=int(ytl)
        xbr=int(xbr)
        ybr=int(ybr)
        flattened = mask.flat[:].tolist()
        flattened.extend([xtl, ytl, xbr, ybr])
        return flattened
    
    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        self.h = h
        self.w=w
        inputs = image
        img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        inp = {self.input_details[0]: im}

        # ONNX inference
        output = list()
        detections = self.model.run(None, inp)

        output0 = detections[0][0].transpose()
        output1 = detections[1][0]
        boxes = output0[:,0:11]
        masks = output0[:,11:]
        output1 = output1.reshape(32,160*160)
        masks = masks @ output1
        boxes = np.hstack((boxes,masks))
        
        yolo_classes = [
            "cherry_tomato", "Mozzarella", "Diced_Ham", "Mushrooms", 
            "Olives", "Salame", "Pepper"
        ]

        objects = []
        for row in boxes:
            prob = row[4:11].max()
            if prob < 0.1:
                continue
            xc,yc,w_,h_ = row[:4]
            class_id = row[4:11].argmax()
            x1 = (xc-w_/2)/640*self.w
            y1 = (yc-h_/2)/640*self.h
            x2 = (xc+w_/2)/640*self.w
            y2 = (yc+h_/2)/640*self.h
            label = yolo_classes[class_id]
            mask = self.get_mask(row[11:25611],(x1,y1,x2,y2))
            mask_h = mask.shape[0]
            mask_w = mask.shape[1]
            cvat_mask = self.to_cvat_mask((x1,y1,x2,y2),mask)
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #all_contour_points = np.vstack(contours)
            #all_contour_points = [(x + x1, y + y1) for x,y in all_contour_points] 
            #epsilon = 0.03 * cv2.arcLength(all_contour_points, True)
            #approx_polygon = cv2.approxPolyDP(all_contour_points, epsilon, True)
            #approx_polygon = np.array([[[p[0][0] + x1, p[0][1] + y1]] for p in approx_polygon])
            approx_polygon = [[contour[0][0],contour[0][1]] for contour in contours[0][0]]
            approx_polygon = np.array([[[p[0] + x1, p[1] + y1]] for p in approx_polygon])
            objects.append([x1,y1,x2,y2,label,prob,cvat_mask,approx_polygon])



        objects.sort(key=lambda x: x[5], reverse=True)
        nms = []
        while len(objects)>0:
            nms.append(objects[0])
            objects = [object for object in objects if self.iou(object,objects[0])<0.8]

        results = []
        for elem in nms:
            results.append({
                        "confidence": str(elem[5]),
                        "label": elem[4],
                        "points": elem[7].ravel().tolist(),
                        "mask": elem[6],
                        "type": "mask",
                    })

        return results
