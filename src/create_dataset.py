from importsfolder import *
from config_file import config

class petFinderDataset:
    def __init__(self, constant_func, dataframe, is_valid = 0):
        self.constant_func = constant_func
        self.dataframe = dataframe
        self.is_valid = is_valid
        if self.is_valid == 1: # transforms for validation images
            self.aug = albumentations.Compose([
            albumentations.RandomResizedCrop(384, 384),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            )], p=1.)

        else:       # transfoms for training images 
            self.aug = albumentations.Compose([
            albumentations.RandomResizedCrop(384, 384),
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
        df = self.dataframe.iloc[item, :]

        # converting jpg format of images to numpy array
        img = np.array(Image.open(self.constant_func.TRAINING_IMAGE_PATH + df["Id"] + '.jpg')) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image = img)['image']
        img = np.transpose(img , (2,0,1)).astype(np.float32) # 2,0,1 because pytorch excepts image channel first then dimension of image
        
        return {
            'image': torch.tensor(img, dtype = torch.float),
            'tabular_data' : torch.tensor(df[1:-2], dtype = torch.float),
            'target' : torch.tensor(df['Pawpularity'], dtype = torch.float)
        }

if __name__ == "__main__":
    new_train = pd.read_csv(config.TRAINING_FILE)
    print(petFinderDataset(config, new_train)[71]['target'])
