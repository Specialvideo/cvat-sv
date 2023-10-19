from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort
import json

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def get_mask(row,box):
    mask = row.reshape(160,160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype('uint8')*1
    x1,y1,x2,y2 = box
    mask_x1 = round(x1/img_width*160)
    mask_y1 = round(y1/img_height*160)
    mask_x2 = round(x2/img_width*160)
    mask_y2 = round(y2/img_height*160)
    mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
    img_mask = Image.fromarray(mask,"L")
    img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
    mask = np.array(img_mask)
    return mask

def sigmoid(z):
    return 1 / (1 + np.exp(-z))




img = Image.open("0.jpg")
img_width, img_height = img.size
device = ort.get_device()
cuda = True if device == 'GPU' else False
try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.log_severity_level = 3

    model = ort.InferenceSession('best.onnx', providers=providers, sess_options=so)
    output_details = [i.name for i in model.get_outputs()]
    input_details = [i.name for i in model.get_inputs()]
except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

image = np.array(img)
image = image[:, :, ::-1].copy()
h, w, _ = image.shape
inputs = image
img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
image = img.copy()
image, ratio, dwdh = letterbox(image, auto=False)
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)

im = image.astype(np.float32)
im /= 255

inp = {input_details[0]: im}
# ONNX inference
output = list()
detections = model.run(None, inp)
output0 = detections[0][0].transpose()
output1 = detections[1][0]
boxes = output0[:,0:11]
masks = output0[:,11:]
output1 = output1.reshape(32,160*160)
masks = masks @ output1
print(masks.shape)
boxes = np.hstack((boxes,masks))

yolo_classes = [
    "cherry_tomato", "Mozzarella", "Diced_Ham", "Mushrooms", 
    "Olives", "Salame", "Pepper"
]

objects = []
for row in boxes:
    prob = row[4:11].max()
    if prob < 0.5:
        continue
    xc,yc,w,h = row[:4]
    class_id = row[4:11].argmax()
    x1 = (xc-w/2)/640*img_width
    y1 = (yc-h/2)/640*img_height
    x2 = (xc+w/2)/640*img_width
    y2 = (yc+h/2)/640*img_height
    label = yolo_classes[class_id]
    mask = get_mask(row[11:25611],(x1,y1,x2,y2))
    objects.append([x1,y1,x2,y2,label,prob,mask])

# apply non-maximum suppression to filter duplicated
# boxes
objects.sort(key=lambda x: x[5], reverse=True)
nms = []
while len(objects)>0:
    nms.append(objects[0])
    objects = [object for object in objects if iou(object,objects[0])<0.5]

results = []
for elem in nms:
    print(elem[6].shape)
    Image.fromarray(elem[6]).show()
    break
    results.append({
                "confidence": str(elem[5]),
                "label": elem[4],
                "points": [elem[0], elem[1], elem[2], elem[3]],
                "type": "rectangle",
            })