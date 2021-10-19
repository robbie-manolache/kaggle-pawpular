
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Module for generating training data batches #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image

class DataBatcher(Sequence):
    """
    """
    
    def __init__(self, mode="train", data_dir=None, 
                 target="pawpularity", x_vars="ALL",
                 batch_size=32, img_size=(224, 224)):
        """
        """
        # get data dir from env if not given
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR")
        
        # set attributes
        self.mode = mode
        self.base_df = pd.read_csv(os.path.join(data_dir, "%s.csv"%mode))
        self.base_df.columns = [c.lower().replace(" ", "_") 
                                for c in self.base_df.columns]
        self.target = target
        if x_vars == "ALL":
            self.x_vars = [x for x in self.base_df.columns 
                           if x not in ["id", target]]
        else:
            self.x_vars = x_vars        
        self.img_dir = os.path.join(data_dir, mode)
        self.img_list = self.base_df["id"].tolist()
        self.img_size = img_size
        self.n_img = self.base_df.shape[0]
        self.batch_size = batch_size
        if len(self.img_list) != self.n_img:
            print("Incosistency detected with number of " +
                  "images in the data vs on disk!")
           
    def __len__(self):
        return int(np.ceil(self.n_img / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = range(self.n_img)
        if self.mode == "train":
            self.indexes = random.sample(self.indexes, 
                                         k=len(self.indexes))
            
    def __getitem__(self, idx):
        
        b = self.batch_size
        batch_images = self.img_list[(idx * b):((1 + idx) * b)]
        
        x_img = np.array([image.img_to_array(
            image.load_img(os.path.join(self.img_dir, bi+".jpg"),
                           target_size=self.img_size)) for bi in batch_images])
        
        x_var = self.base_df.query("id in @batch_images")[self.x_vars].values
        y_var = self.base_df.query("id in @batch_images")[self.target].values
        
        return [x_img, x_var], y_var         
        