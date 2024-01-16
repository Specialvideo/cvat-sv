from PIL import Image
import os, torch, cv2
import numpy as np
from torch import nn
#from utils import compute_metrics
import numpy as np
import base64
from torchvision import transforms
from transformers import AutoImageProcessor
from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation
import pickle
class ModelHandler:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.load_network(checkpoint='b4_checkpoints/checkpoint-400')
        self.labels = ['bg','olive', 'anchovy', 'red_pepper','salami', 'yellow_pepper']
        self.id2label = {int(k): v for k,v in enumerate(self.labels)}
        self.label2id = {v: k for k,v in self.id2label.items()}


    def rle( ar ):
        i =0
        s = ""
        while i<len(ar):
            c=0
            el = ar[i]
            while i<len(ar) and  abs(ar[i]-el)<=10 :
                c+=1
                i+=1
            
            el_bin = "{0:08b}".format(el)
            c_bin = "{0:08b}".format(c)
            s = s + el_bin + c_bin
        return s

    def load_network(self, checkpoint):
        self.image_processor = AutoImageProcessor.from_pretrained('nvidia/mit-b4')

        #TODO mettere pesi fine tunati pizze
        self.model = AutoModelForSemanticSegmentation.from_pretrained('b4_checkpoints/checkpoint-400')#.to('cuda')
        
    
    def infer(self, image, context):
        context.logger.info(f"\n\nDENTRO INFER\n\n")
        classes = [
            "Anchovy", "Olives", "Salame", "Red_Pepper", 
            "Yellow_Pepper"
        ]
        image = np.array(image).astype(np.uint8)
        context.logger.info(f"input shape:{image.shape}")
        
        encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values#.to('cuda')
        
        output = self.model(pixel_values=pixel_values)
        
        logits = output.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(image.shape[0],image.shape[1]),
            mode="bilinear",
            align_corners=False,
        ) 
        objects=[]
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        context.logger.info(f"pred_seg shape:{pred_seg.shape}")
        for value in np.unique(pred_seg):
            #exluding background
            if value!=0:
                black_img = np.zeros_like(pred_seg)
                black_img[pred_seg == value] = value
                output = cv2.connectedComponentsWithStats(black_img.astype(np.uint8), connectivity=8)
                (numLabels, labels, stats, centroids) = output
                for i in range(1, numLabels):
                    componentMask = (labels == i).astype("uint8") * 255
                    contours = cv2.findContours(componentMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
                    if len(contours[0])>2:
                        approx_polygon = [[contour[0][0],contour[0][1]] for contour in contours[0]]
                        approx_polygon = np.array([[[p[0], p[1]]] for p in approx_polygon])
                        context.logger.info(f"i:{i}")
                        objects.append([approx_polygon,classes[value-1]])

        
        results = []
        for i,elem in enumerate(objects):
            results.append({
                        "confidence":str(1),
                        "label": elem[1],
                        "points": elem[0].ravel().tolist(),
                        "type":"mask",
                    })
        
        return results
                    
        
        
