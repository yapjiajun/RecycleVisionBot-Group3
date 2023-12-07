#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:14:13 2023

@author: zxc703
"""

import coppeliasim_zmqremoteapi_client as zmq
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import ecse275_vision_utils as util
import my_functions as func
import time

'''PASTE YOUR NEURAL NETWORKS HERE'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dim = 512 * 384 * 3
        self.fc1 = nn.Linear(self.dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = x.view(-1, self.dim)  # Flatten the input
        x = F.relu(self.fc1(x))  # ReLU Linear Layer
        x = F.relu(self.fc2(x))  # ReLU Linear Layer
        x = self.fc3(x)  # Linear Layer
        return x



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.pool4 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(256, 32) #Changed to match shape
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) #Added to reshape data properly
        x = self.pool1(F.relu(self.conv1(x))) # Conv ReLU Pool
        x = self.pool2(F.relu(self.conv2(x))) # Conv ReLU Pool
        x = self.pool3(F.relu(self.conv3(x))) # Conv ReLU Pool
        x = self.pool3(F.relu(self.conv3(x))) # Conv ReLU Pool
        x = self.pool3(F.relu(self.conv3(x))) # Conv ReLU Pool
        x = self.pool3(F.relu(self.conv3(x))) # Conv ReLU Pool
        x = self.pool4(F.relu(self.conv4(x))) # Conv ReLU Pool
        x = x.reshape(-1, 256) #Changed from "x.view(-1, 256)"
        x = F.relu(self.fc1(x)) # ReLU Linear Layer
        x = self.fc2(x) # Linear Layer
        return x



plt.close("all")

cats = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  
f = 0.020 # focal length in meters
pixels_per_inch = 560.0165995731867 
z = 0.805 # vertical distance to the centerpoint of the blocks on the table

vision_mode = "RGB" # or "RGB"


client = zmq.RemoteAPIClient()
sim = client.getObject('sim')

drop_CB = sim.getObject('/drop_CB')
drop_G = sim.getObject('/drop_G')
drop_MC = sim.getObject('/drop_MC')
drop_PA = sim.getObject('/drop_PA')
drop_PL = sim.getObject('/drop_PL')
drop_T = sim.getObject('/drop_T')
dropptCB = sim.getObjectPose(drop_CB,-1)
dropptG = sim.getObjectPose(drop_G,-1)
dropptMC = sim.getObjectPose(drop_MC,-1)
dropptPA = sim.getObjectPose(drop_PA,-1)
dropptPL = sim.getObjectPose(drop_PL,-1)
dropptT = sim.getObjectPose(drop_T,-1)


camera_handle = sim.getObject("/Vision_sensor")

prox1 = sim.getObject("/conveyor/Proximity_sensor")


while True:
    time.sleep(3)
    result = sim.readProximitySensor(prox1)
    
    if result[0] > 0:
        if vision_mode == "gray":
            image,resolution = sim.getVisionSensorImg(camera_handle,1)
            image = np.array(bytearray(image),dtype='uint8').reshape(resolution[0],resolution[1]) 
        elif vision_mode == "RGB":
            image,resolution = sim.getVisionSensorImg(camera_handle,0)
            igrey,resolution2 = sim.getVisionSensorImg(camera_handle,1)
            image = np.array(bytearray(image),dtype='uint8').reshape(resolution[0],resolution[1],3)
            igrey = np.array(bytearray(igrey),dtype='uint8').reshape(resolution2[0],resolution2[1]) 
        else:
            print("invalid!")
        
        image = np.flip(image,axis=1)
        igrey = np.flip(igrey,axis=1)
        print(image)
        if vision_mode == "gray":
            plt.imshow(image,cmap="binary")
        elif vision_mode =="RGB":
            plt.imshow(image)
            plt.show
    
        
        centroids,list_of_cube_images = util.process_image(igrey,image)
        
        print(list_of_cube_images)
        for i in range(len(list_of_cube_images)):
            plt.figure()
            plt.imshow(list_of_cube_images[i],cmap="binary")
        
        model_type = "FCN100" # specify which model you would like here. Choices are FCN, CNN, CNNaug
        
        if model_type=="FCN":
            net = Net()
            net.load_state_dict(torch.load("fully_connected.pth"))
        elif model_type=="FCN100":
            net = Net()
            net.load_state_dict(torch.load("fully_connected_e100.pth"))
        elif model_type=="CNN":
            net = CNN()
            net.load_state_dict(torch.load("convolutional.pth"))
        
        predictions = []
        for i in range(len(list_of_cube_images)):
            predictions.append(cats[torch.argmax(net(torch.Tensor(list_of_cube_images[i]).unsqueeze(0))).numpy()])
        print("identified trash is:")
        print(np.array(predictions))
        
        order_sequence = np.argsort(predictions) #sort by descending order
        
            
        #%%
        T_cam_world = np.array(sim.getObjectMatrix(camera_handle,-1)).reshape(3,4)
        pos_cam_list = []
        pos_world_list = []
        for i in range(len(predictions)):
            pos_cam_list.append(func.compute_pos_from_pix(centroids[i],resolution,f,pixels_per_inch,z))
        for i in range(len(pos_cam_list)):
            pos_world_list.append(util.hand_eye_transform(pos_cam_list[i],T_cam_world))
        
        #%% Movement Commands
        
        for i in range(len(pos_world_list)):
            print("picking..." + str(predictions[i]))
            pickPoint = list(pos_world_list[i])
            util.move_to(sim,[pickPoint[0],pickPoint[1],pickPoint[2]],offset=0.02)
            util.toggle_gripper(sim)
            util.move_to(sim,[pickPoint[0],pickPoint[1],pickPoint[2]+0.05])
            
            match predictions[i]:
                
                case "cardboard":
                    droppt = dropptCB
                case  'glass' :
                    droppt = dropptG
                case 'metal' :
                    droppt = dropptMC
                case 'paper' :
                    droppt = dropptPA
                case 'plastic' :
                    droppt = dropptPL
                case 'trash' :
                    droppt = droppt
     
            util.move_to(sim,droppt,offset=0.02)
            util.toggle_gripper(sim)
            reset_point = [droppt[0],droppt[1],droppt[2]+0.1]
            util.move_to(sim,reset_point,approach_height=0)
                
