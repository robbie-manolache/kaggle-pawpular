
# <<<<<<<<<<<<<<<<<<<<<<<<<< #
# Feature Engineering Module #
# >>>>>>>>>>>>>>>>>>>>>>>>>> #

import os
import numpy as np
import cv2

def compute_blur(img, norm=100):
    """
    """
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(img_bw, cv2.CV_64F).std()
    return blur / norm

def compute_img_stats(img_id, img_dir):
    """
    """
    
    # load image data
    img_path = os.path.join(img_dir, img_id + ".jpg")
    img0 = cv2.imread(img_path)
    
    # aspect ratio and orientation
    aspect_ratio = img0.shape[0]/img0.shape[1]
    portrait = int(aspect_ratio > 1)
    if portrait == 1:
        aspect_ratio = 1/aspect_ratio
        
    # RGB channel stats
    r_mu, g_mu, b_mu = img0.mean(axis=(0,1))/255
    r_sd, g_sd, b_sd = img0.std(axis=(0,1))/255
    
    # Return all stats
    return {
        "id": img_id,
        "portrait": portrait,
        "aspect_ratio": aspect_ratio,
        "resolution": np.sqrt(img0.shape[0]*img0.shape[1])/1280,
        "blur_level": compute_blur(img0),
        "r_mean": r_mu,
        "r_stdev": r_sd,
        "g_mean": g_mu,
        "g_stdev": g_sd,
        "b_mean": b_mu,
        "b_stdev": b_sd
    }
    
