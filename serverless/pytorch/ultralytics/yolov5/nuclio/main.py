"""

import json
import base64
from PIL import Image
import io
import torch
from model_handler import ModelHandler
def init_context(context):
    print('BBBBB\n\n')
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    print('AAAAAAAA\n\n')
    context.logger.info("Run yolo-v5 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
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
"""
"""
import json
import base64
from PIL import Image
import io
import torch

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = torch.hub.load('/opt/nuclio/yolov5', "custom", path='/opt/nuclio/yolov5l.pt', source="local")
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v5 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")
    predictions = context.user_data.model_handler(image)

    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    results = []
    for box, score, label in zip(pred_boxes, scores, pred_classes):
        label = COCO_CATEGORIES[int(label)]["name"]
        if score >= threshold:
            results.append({
                "confidence": str(float(score)),
                "label": label,
                "points": box.tolist(),
                "type": "rectangle",
            })
"""
import json
import base64
from PIL import Image
import io
import torch

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = torch.hub.load('/opt/nuclio/yolov5', "custom", path='/opt/nuclio/yolov5l.pt', source="local")
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
            
            
            

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
