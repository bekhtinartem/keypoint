import torch
import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FaceKeypointResNet50
model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load('F:/key_points/model_1.pth', map_location='cpu')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
t_p=[]
t_x=[]
t_y=[]
f=open("F:/key_points/for_test.csv", 'r')

for s in f:
    s=s[1:]
    name=s[0:s.find(";")]
    s=s[s.find(";")+1:]
    k=s.split(";")
    for i in range(26):
        k.pop()
    point1=[]
    x1=[]
    y1=[]
    for i in range(len(k)):
        if(i%3==0):
           point1.append(float(k[i]))
        if (i % 3 == 1):
            x1.append(float(k[i]))
        if (i % 3 == 2):
            y1.append(float(k[i]))
    t_p.append(point1)
    t_x.append(x1)
    t_y.append(y1)
    id=("F:/Dataset/data/cow/" + name[name.find("c"):])
    print(id)
    cap = cv2.VideoCapture(id)



    for i in range(1):
        # capture each frame of the video
        ret, frame = cap.read()
        if True:
            with torch.no_grad():
                image = frame
                image = cv2.resize(image, (224, 224))
                plt.show()
                orig_frame = image.copy()

                orig_h, orig_w, c = orig_frame.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0
                image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image, dtype=torch.float)
                image = image.unsqueeze(0).to(config.DEVICE)
                outputs = model(image)
                #print(outputs)
                count=0
                point=[]
                x=[]
                y=[]

                for i in outputs:
                    for j in i:
                        #print(str(j)[7:len(str(j))-1])
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
            for i in range(len(point1)):
                if (point1[i] > 0.5):
                    orig_frame = cv2.circle(orig_frame, (int(x1[i]), int(y1[i])), radius=8,
                                            color=(250, 0, 0), thickness=-1)
            for i in range(len(point)):
                if(point[i]>0.5):
                    orig_frame = cv2.circle(orig_frame, (int(x[i]*1024),int(y[i]*683)), radius=8, color=(0, 0, 250), thickness=-1)

            plt.imshow(orig_frame)
            plt.show()

