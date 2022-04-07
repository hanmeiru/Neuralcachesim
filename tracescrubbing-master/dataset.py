import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CacheDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) #list files in the folder image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("L"), dtype = np.float32) #L for greyscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32) #, dtype = np.float(32)--need to change dtype? 
    

        mask[mask == 255.0] = 1.0 #for sigmoid (?)-- about probability  

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) #?? 
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask = mask[None,:] #adding one dimension

        return image, mask