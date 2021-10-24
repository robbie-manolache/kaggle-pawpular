
# <<<<<<<<<<<<<<<<<<<<<<< #
# Module for loading data #
# >>>>>>>>>>>>>>>>>>>>>>> #

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

def __gen_target_df__(df):
    """
    """
    tgt_df = df.copy()[["id", "pawpularity"]]
    tgt_df.loc[:, "id"] = tgt_df["id"] + ".jpg"
    
    return tgt_df

class DataLoader:
    
    def __init__(self, mode="train", data_dir=None):
        """
        """
        
        # get data dir from env if not given
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR")
        
        # set attributes
        self.data_dir = data_dir
        self.base_df = pd.read_csv(os.path.join(data_dir, "%s.csv"%mode))
        self.base_df.columns = [c.lower().replace(" ", "_") 
                                for c in self.base_df.columns]
        self.target_df = __gen_target_df__(self.base_df)
        self.img_dir = os.path.join(data_dir, mode)
        self.n_img = self.target_df.shape[0]
    
    def valid_split(self, valid_frac=0.1, seed=42):
        """
        """
        self.train_df, self.valid_df = train_test_split(self.base_df,
                                                        test_size=valid_frac,
                                                        random_state=seed)
        
    def image_df_batcher(self, valid_frac=0.1, pre_proc_args=None,
                         target_pix=224, batch_size=5, sub_sample=None):
        """
        """
        
        data_gen = image.ImageDataGenerator(
            validation_split=valid_frac,
            **pre_proc_args
        )
        
        if sub_sample is None:
            flow_df = self.target_df
        else:
            if sub_sample < 1:
                sub_sample = int(np.round(sub_sample*self.n_img))
            flow_df = self.target_df.sample(sub_sample)
        
        flow_args = {
            "dataframe": flow_df,
            "directory": self.img_dir,
            "x_col": "id",
            "y_col": "pawpularity",
            "batch_size": batch_size,
            "target_size": (target_pix, target_pix),
            "class_mode": "raw",
            "shuffle": True,
            "seed": 42
        }
        
        if valid_frac > 0:
            return (data_gen.flow_from_dataframe(**flow_args, subset="training"),
                    data_gen.flow_from_dataframe(**flow_args, subset="validation"))
        else:
            return data_gen.flow_from_dataframe(**flow_args)
        