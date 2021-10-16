
# <<<<<<<<<<<<<<<<<<<<<<<< #
# Module for visualization #
# >>>>>>>>>>>>>>>>>>>>>>>> #

import numpy as np
import matplotlib.pyplot as plt

def img_gen_subplots(img_gen, n_runs=5):
    
    plt.figure(figsize=(img_gen.batch_size*2.5, n_runs*2.5))
    a = 0
    
    for r in range(n_runs):
    
        x = img_gen.next()
        
        for i in range(x[0].shape[0]):

            a += 1
            
            plt.subplot(n_runs, img_gen.batch_size, a)
            plt.imshow(x[0][i, :, :, :].astype(np.uint8))
            plt.axis("off")
            plt.title("Score: %d"%x[1][i])
