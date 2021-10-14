
# <<<<<<<<<<<<<<<<<<<<<<< #
# Module for loading data #
# >>>>>>>>>>>>>>>>>>>>>>> #

import os
import pandas as pd

def __gen_target_df__(df):
    """
    """
    tgt_df = df.copy()[["id", "pawpularity"]]
    tgt_df.columns = ["id", "label"]
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
        self.img_set = None
        
        
        
