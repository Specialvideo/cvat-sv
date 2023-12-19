import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image
import base64
from skimage.measure import approximate_polygon, find_contours
import supervision as sv
import torch
from segment_anything import sam_model_registry, SamPredictor
class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolox_200.pt")
        self.labels = labels
        self.h=0
        self.w=0

    def load_network(self, model):
        device = ort.get_device()
        self.model = YOLO(model)

  
    
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
        #YOLO inference
        
        results = self.model([image], stream=True)
        
        yolo_classes = [
            "Olives", "Anchovy", "Salame", "Red_Pepper", 
            "Yellow_Pepper"
        ]
        
        #definition of SAM
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        SAM_ENCODER_VERSION = "vit_h"
        
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint='sam_vit_h.pth')
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        
        sam_predictor.set_image(np.array(image))
        
        objects = []
        for row in results:
            for box in row.boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 =x1.item(),y1.item(),x2.item(),y2.item()
                label = yolo_classes[int(box.cls[0])]

                input_box = np.array([x1,y1,x2,y2])
                m, score, logit = sam_predictor.predict(point_coords=None,
                                                            point_labels=None,
                                                            box=input_box[None,:],
                                                            multimask_output=False)

                mask = m[0].astype(np.uint8) * 255

                cvat_mask = self.to_cvat_mask((x1,y1,x2,y2),m[0])

                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
                if len(contours[0])>2:
                    approx_polygon = [[contour[0][0],contour[0][1]] for contour in contours[0]]
                    approx_polygon = np.array([[[p[0], p[1]]] for p in approx_polygon])

                    objects.append([x1,y1,x2,y2,label,0.1,cvat_mask,approx_polygon])
        results = []
        for elem in objects:
            results.append({
                        "confidence": str(elem[5]),
                        "label": elem[4],
                        "points": elem[7].ravel().tolist(),
                        "mask": elem[6],
                        "type": "mask",
                    })

        return results
