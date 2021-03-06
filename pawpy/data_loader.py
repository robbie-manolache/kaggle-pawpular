
# <<<<<<<<<<<<<<<<<<<<<<< #
# Module for loading data #
# >>>>>>>>>>>>>>>>>>>>>>> #

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from pawpy.feat_eng import compute_img_stats

def __load_base_df__(data_dir, mode):
    """
    """
    df = pd.read_csv(os.path.join(data_dir, "%s.csv"%mode))
    if mode == "test":
        df.loc[:, "pawpularity"] = -1    
        
    return df

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
        self.base_df = __load_base_df__(data_dir, mode)
        self.base_df.columns = [c.lower().replace(" ", "_") 
                                for c in self.base_df.columns]
        self.target_df = __gen_target_df__(self.base_df)
        self.img_dir = os.path.join(data_dir, mode)
        self.n_img = self.target_df.shape[0]
    
    def gen_extra_features(self, load_path=None, save_path=None):
        """
        """
        
        if load_path is None:
            img_stats = []
            for img_id in tqdm(self.base_df["id"].values):
                img_stats.append(compute_img_stats(img_id, self.img_dir))
            new_df = pd.DataFrame(img_stats)
        else:
            new_df = pd.read_csv(load_path)
        
        if save_path is not None:
            new_df.to_csv(save_path, index=False)
        
        self.base_df = self.base_df.merge(new_df, on="id")
    
    def valid_split(self, valid_frac=0.1, seed=42):
        """
        """
        self.train_df, self.valid_df = train_test_split(self.base_df,
                                                        test_size=valid_frac,
                                                        random_state=seed)
        
    def image_df_batcher(self, valid_frac=0.1, pre_proc_args=None,
                         target_pix=224, batch_size=5, 
                         sub_sample=None, seed=42):
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
            flow_df = self.target_df.sample(sub_sample, random_state=seed)
        
        flow_args = {
            "dataframe": flow_df,
            "directory": self.img_dir,
            "x_col": "id",
            "y_col": "pawpularity",
            "batch_size": batch_size,
            "target_size": (target_pix, target_pix),
            "class_mode": "raw",
            "shuffle": True,
            "seed": seed
        }
        
        if valid_frac > 0:
            return (data_gen.flow_from_dataframe(**flow_args, subset="training"),
                    data_gen.flow_from_dataframe(**flow_args, subset="validation"))
        else:
            return data_gen.flow_from_dataframe(**flow_args)
        