# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
"""
import numpy as np
import cv2
import torch

def convert_mask_to_polygon(mask):
    print('CCCCCCC\n\n')
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon

class ModelHandler:
    def __init__(self):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolov5_checkpoint = "/opt/nuclio/yolov5/yolov5s.pt"
        self.model_type = "5x"
        self.latest_image = None
        self.yolov5_model = torch.hub.load("ultralytics/yolov5","YOLOv5s")
        #yolov5_model.to(device=self.device)

    def handle(self, image):
        print('DDDDDDD\n\n')
        res = self.yolov5_model(np.array(image))
        #features = self.predictor.get_image_embedding()
        return res
"""       
        
        
import json
import base64
from PIL import Image
import io
import torch

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = torch.hub.load('/opt/nuclio/yolov5', "custom", path='/opt/nuclio/yolov5s.pt', source="local")
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v5 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    yolo_results_json = context.user_data.model(image).pandas().xyxy[0].to_dict(orient='records')

    encoded_results = []
    for result in yolo_results_json:
        encoded_results.append({
            'confidence': result['confidence'],
            'label': result['name'],
            'points': [
                result['xmin'],
                result['ymin'],
                result['xmax'],
                result['ymax']
            ],
            'type': 'rectangle'
        })

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)
