from unicodedata import category
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import os
from train import get_model
import gdown

def get_prediction(img, threshold):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img.to(device)])
    pred_class = [category_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
    except:
        None
    return pred_boxes, pred_class
  


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    boxes, pred_cls = get_prediction(img, threshold)
    img = np.asarray(img)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i].capitalize(), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    return img



if __name__ == "__main__":

    url = "https://drive.google.com/file/d/1NDbgawtmajcrqof0GjV3ZFcfwBJdfIER/view?usp=sharing"
    output = 'model.pth'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 4

    model = get_model(num_classes)

    model = torch.load('model.pth') 
    model.to(device)
    model.eval()
    
    category_names = ['__background__', 'water', 'soda', 'juice']
    
    stream = cv2.VideoCapture(0)
    
    while True:
        ret, frame = stream.read()
        frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)

        image = Image.fromarray(frame).convert("RGB")

        img = object_detection_api(image, threshold=0.9)
        cv2.imshow('Demo', img)

        c = cv2.waitKey(1)
        if c == 27:
            break

    stream.release()
    cv2.destroyAllWindows()