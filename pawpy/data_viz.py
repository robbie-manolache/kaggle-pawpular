
# <<<<<<<<<<<<<<<<<<<<<<<< #
# Module for visualization #
# >>>>>>>>>>>>>>>>>>>>>>>> #

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import decode_predictions

def img_gen_subplots(img_gen, n_runs=5, model=None, model_type="eff_net"):
    """
    Later we'll pass predictions from own models to compare with actual scores!
    """
    
    plt.figure(figsize=(img_gen.batch_size*4, n_runs*4))
    a = 0
    
    for r in range(n_runs):
    
        x = img_gen.next()
        if model is not None:
            if model_type == "eff_net":
                preds = decode_predictions(model.predict(x))
        
        for i in range(x[0].shape[0]):

            a += 1
            if model is not None:
                if model_type == "eff_net":
                    class_id, name, prob = preds[i][0]
            
            plt.subplot(n_runs, img_gen.batch_size, a)
            plt.imshow(x[0][i, :, :, :].astype(np.uint8))
            plt.axis("off")
            if model is None:
                plt.title("Score: %d"%x[1][i])
            else:
                if model_type == "eff_net":
                    plt.title("%s (%.1f%%) - %d"%(name, prob*100, x[1][i]))
