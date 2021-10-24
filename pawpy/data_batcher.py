
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Module for generating training data batches #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image

def __load_aug_img__(path, img_size, img_aug):
    """
    """
    img = image.img_to_array(image.load_img(path, target_size=img_size))
    if img_aug is not None:
        img = img_aug.random_transform(img)
    return img

class DataBatcher(Sequence):
    """
    img_aug:    tensorflow.keras.preprocessing.image.ImageDataGenerator
    """
    
    def __init__(self, df, img_dir, mode="train", batch_size=32, 
                 target="pawpularity", x_vars="ALL",
                 img_size=(224, 224), img_aug=None, seed=42):
        """
        """
        self.mode = mode
        self.img_df = df
        self.target = target
        if x_vars == "ALL":
            self.x_vars = [x for x in self.img_df.columns 
                           if x not in ["id", target]]
        else:
            self.x_vars = x_vars        
        self.img_dir = img_dir
        self.img_list = self.img_df["id"].tolist()
        self.img_size = img_size
        self.img_aug = img_aug
        self.n_img = self.img_df.shape[0]
        self.batch_size = batch_size
        self.seed = seed
        if self.mode == "train":
            np.random.seed(self.seed)
           
    def __len__(self):
        return int(np.ceil(self.n_img / self.batch_size))
      
    def on_epoch_end(self):
        self.indexes = range(self.n_img)
        if self.mode == "train":
            self.seed += 1
            np.random.seed(self.seed)
            self.indexes = random.sample(self.indexes, 
                                         k=len(self.indexes))
            
    def __getitem__(self, idx):
        
        b = self.batch_size
        batch_images = self.img_list[(idx * b):((1 + idx) * b)]
        
        x_img = np.array([__load_aug_img__(
            os.path.join(self.img_dir, bi+".jpg"), self.img_size, 
            self.img_aug) for bi in batch_images])
        
        x_var = self.img_df.query("id in @batch_images")[self.x_vars].values
        y_var = self.img_df.query("id in @batch_images")[self.target].values
        
        return [x_img, x_var], y_var         
        