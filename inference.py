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
from tqdm import tqdm
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
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size= skip_connection.shape[2:]) 
            
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+ 1](concat_skip)
        
        return self.final_conv(x)

def model_fn(model_dir):
    print("In model_fn. Model directory is - ")
    print(model_dir)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    with open("./model.pth", "rb") as f:
        print("Loading the U-NET model")
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully!\n')
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    test_transform = transforms.Compose([transforms.Resize((160, 240)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                              std=[1.0, 1.0, 1.0])])
    input_object=test_transform(input_object) #1st error with Image # Solved
    input_object = input_object.cuda() #put data into GPU
    with torch.no_grad():
        input_object = input_object.unsqueeze(0)
        preds = torch.sigmoid(model(input_object)) #2nd error #Solved
        prediction = (preds > 0.5).float()
    return prediction

def main():
    model_dir = input("Please input your model path ")# Model path and name
    model = model_fn(model_dir)
    image_folder = input("\nPlease input your image folder path ")# test image folder
    saved_folder = input("\nPlease input your saving folder path ") #prediction result folder
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    count = os.listdir(image_folder)
    for i in tqdm(range(len(count)-1), desc = " Segmentating your images! Progress: "):
        img_name = count[i]
        with open(os.path.join(image_folder, img_name),"rb") as myimg:
            payload = myimg.read() 
        imgf = input_fn(payload, content_type=JPEG_CONTENT_TYPE)
        prediction = predict_fn(imgf, model)
        torchvision.utils.save_image(
        prediction,  str(saved_folder) +"/" + str(img_name[:-4]) + "_pred.png")
    logger.info("DONE!!! Please check your masks in " + saved_folder)
        
if __name__ == "__main__":
    main()
