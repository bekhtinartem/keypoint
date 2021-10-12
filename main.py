import torch
import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FaceKeypointResNet50
model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load('F:/key_points/model.pth', map_location='cpu')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# capture the webcam
cap = cv2.VideoCapture("F:/key_points/cow_11244.jpg")



for i in range(1):
    # capture each frame of the video
    ret, frame = cap.read()
    if True:
        with torch.no_grad():
            image = frame
            image = cv2.resize(image, (224, 224))
            cv2.imshow("1", image)
            plt.imshow(image)
            plt.show()
            orig_frame = image.copy()

            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)
            print(outputs)
            count=0
            point=[]
            x=[]
            y=[]

            for i in outputs:
                for j in i:
                    print(str(j)[7:len(str(j))-1])
                    if count%3==0:
                        point.append(float(str(j)[7:len(str(j))-1]))
                    if count%3==1:
                        x.append(float(str(j)[7:len(str(j))-1]))
                    if count %3==2:
                        y.append(float(str(j)[7:len(str(j))-1]))
                    count+=1


        print(x)
        print(y)
        outputs = outputs.cpu().detach().numpy()
        #outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        frame_width = 1024
        frame_height = 683
        orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
        for i in range(len(point)):
            if(point[i]>0.5):
                orig_frame = cv2.circle(orig_frame, (int(x[i]),int(y[i])), radius=5, color=(0, 0, 255), thickness=-1)

        plt.imshow(orig_frame)
        plt.show()

