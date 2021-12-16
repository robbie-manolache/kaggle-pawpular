
import os
from pawpy import env_config
from lazykaggler import kernel_output_download
from datetime import datetime

# set config and data directory
env_config("config.json")
user = "slashie"
kernels = ["ppc-train-nn-0%d"%i for i in [1,2,3,4,5,6,7,8]]
local_dir = os.path.join(os.environ.get("DATA_DIR"), 
                         "kaggle-kernel-download")

# run download
for kernel in kernels:
    now = datetime.now().strftime("_%Y%m%d_%H%M%S")
    local_path = os.path.join(local_dir, kernel+now)
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    kernel_output_download(user, kernel, local_path)
    