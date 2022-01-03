import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512],):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size= 2, stride = 2,
                )
            ) 
            self.ups.append(DoubleConv(feature * 2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size= 1)
     
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        # reverse the skipconnection list
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                # take out the width and heigh of neuron, we use only channel
                # eg. 161 x 161 -> output: 160 x 160 (we can't pooling for edges)
                x = TF.resize(x, size= skip_connection.shape[2:]) 
            
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+ 1](concat_skip)
        
        return self.final_conv(x)

def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the U-NET model")
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    #model.load_state_dict(checkpoint)
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    
    logger.info('In predict fn')
    test_transform = transforms.Compose([transforms.Resize((160, 240)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                              std=[1.0, 1.0, 1.0])])
                                                              #max_pixel_value=255.0,
    logger.info("transforming input")
    input_object=test_transform(input_object) #1st error with Image # Solved
    input_object = input_object.cuda() #put data into GPU
    with torch.no_grad():
        input_object = input_object.unsqueeze(0)
        logger.info("Calling model")
        preds = torch.sigmoid(model(input_object)) #2nd error #Solved
        preds = (preds > 0.5).float()
        prediction = preds
    logger.info(prediction)
    torchvision.utils.save_image(
            preds, f"./pred_.png"
        )
    return prediction