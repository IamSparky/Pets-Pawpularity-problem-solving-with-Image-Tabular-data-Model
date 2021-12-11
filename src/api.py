import os
import sys
import torch
import albumentations
import numpy as np

from flask import Flask
from flask import render_template
from flask import request
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append("pip_installs_required/timm_master/")
import timm

app = Flask(__name__)
UPLOAD_FOLDER = "static"
DEVICE = torch.device("cuda")

class TimmEfficientNet_b0(nn.Module):
    def __init__(self):
        super(TimmEfficientNet_b0, self).__init__()
        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128 + 12, 1)
        
    def forward(self, image, tabular_data_inputs):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, tabular_data_inputs], dim=1)
        x = self.out(x)
        
        return x

class petFinderDataset:
    def __init__(self, constant_func, dataframe):
        self.constant_func = constant_func
        self.dataframe = dataframe
        self.aug = albumentations.Compose([
            albumentations.RandomResizedCrop(224, 224),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)], p=1.)
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        # converting jpg format of images to numpy array
        img = np.array(Image.open(self.constant_func)) 
        img = self.aug(image = img)['image']
        img = np.transpose(img , (2,0,1)).astype(np.float32) # 2,0,1 because pytorch excepts image channel first then dimension of image
    
        return {
            'image': torch.tensor(img, dtype = torch.float),
            'tabular_data' : torch.tensor(self.dataframe, dtype = torch.float)
        }

def Pawpularity_prediction(image_path, arr_data):
    dataset_dict = petFinderDataset(image_path, arr_data)

    test_dataloader = DataLoader(dataset_dict,
                                num_workers=0,
                                batch_size=1,
                                shuffle=False)

    model = TimmEfficientNet_b0()
    model = model.to(DEVICE)
    checkpoint_path = "trained_models/second_timm_efficientnet_b0_ns_model.bin"
    model = torch.load(checkpoint_path)
    model.eval()

    final_preds = None

    for i in range(5):
        for batch_index,dataset in enumerate(test_dataloader):
            tabular_data = dataset['tabular_data']
            image = dataset['image']
            
            tabular_data = tabular_data.to(DEVICE, dtype=torch.float)
            image = image.to(DEVICE, dtype=torch.float)

            with torch.no_grad():
                preds = model(image, tabular_data)
            temp_preds = preds.detach().cpu().numpy()

        if i == 0:
            final_preds = temp_preds
        else:
            final_preds += temp_preds
            
    return final_preds/5

@app.route("/", methods = ["GET", "POST"])
def upolad_predict():
    if request.method == "POST":

        image_file = request.files["image"]
        sub_focus = request.form.get('sub_focus')
        eyes = request.form.get('eyes')
        face = request.form.get('face')
        near = request.form.get('near')
        action = request.form.get('action')
        accessory = request.form.get('accessory')
        group = request.form.get('group')
        collage = request.form.get('collage')
        human = request.form.get('human')
        occlusion = request.form.get('occlusion')
        info = request.form.get('info')
        blur = request.form.get('blur')

        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
                )

            image_file.save(image_location)
            arr_data = [sub_focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur]
            pred_score = Pawpularity_prediction(image_location, list(map(int, arr_data)))
            
            return render_template('index.html', Pawpularity = pred_score[0][0])

    return render_template('index.html', Pawpularity = 0)

if __name__ == "__main__":
    app.run(port = 12000, debug = True)