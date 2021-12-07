
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Interpretability toolset for Neural Networks #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def nn_interpreter(model, config):
    """
    """
    
    # retrieve config items
    params = config["params"]
    x_vars = config["tab_feats"]
    
    # get layer names
    layer_names = [m.name for m in model.layers]
    
    # if Conv2D used to post-process images
    if "post_conv2D" in layer_names: 
        
        # retrieve filter weights    
        post_conv_wb = model.get_layer("post_conv2D").get_weights()
        vmin, vmax = post_conv_wb[0].min(), post_conv_wb[0].max()

        # init plot for each filter
        print("Conv2D Filter Weights:")
        fig, ax = plt.subplots(1, params["conv"]["nf"], 
                               figsize=(params["conv"]["nf"]*7, 7))

        # plot each filter's weights as a heatmap
        for i in range(params["conv"]["nf"]):
            if params["conv"]["nf"] == 1:
                ax = [ax]
            mat = post_conv_wb[0][:, :, 0, i]
            sns.heatmap(mat, ax=ax[i], cmap="coolwarm", annot=True,
                        vmin=vmin, vmax=vmax, cbar=False)
            ax[i].axes.xaxis.set_ticks([])
            ax[i].axes.yaxis.set_ticks([])
            
        plt.show()
    
    # get weights from final layer    
    final_layer_wb = model.get_layer("final_layer").get_weights()

    # if extra Dense layer is used
    if "extra_dense" in layer_names:
        
        extra_dense_wb = model.get_layer("extra_dense").get_weights()

        print("Average weight and st. dev. in extra dense layer:")
        print(pd.DataFrame({"mean": extra_dense_wb[0].mean(axis=0), 
                            "st-dev": extra_dense_wb[0].std(axis=0),
                            "bias": extra_dense_wb[1],
                            "weight": final_layer_wb[0][:, 0]}))

        print("Cross-correlation of extra dense layer weights:")
        print(pd.DataFrame(extra_dense_wb[0]).corr())
        
        print("Now showing feature weights in extra dense layer:")
        wgt_mat = extra_dense_wb[0].transpose()
        
    else:
        
        print("Average weight and st. dev. in final layer:")
        print(pd.DataFrame({"mean": final_layer_wb[0].mean(axis=0), 
                            "st-dev": final_layer_wb[0].std(axis=0),
                            "bias": final_layer_wb[1]}))
        
        print("Now showing feature weights in final layer:")
        wgt_mat = final_layer_wb[0].transpose()
        

    fig, ax = plt.subplots(1, 1, figsize=(25, wgt_mat.shape[0]))
    sns.heatmap(wgt_mat, ax=ax, cmap="coolwarm")
    plt.axis("off")
    plt.show()

    print("Metadata feature weights only:")
    meta_df = pd.DataFrame(wgt_mat[:, -len(x_vars):])
    meta_df.columns = x_vars
    fig, ax = plt.subplots(1, 1, figsize=(len(x_vars), wgt_mat.shape[0]))
    sns.heatmap(meta_df, ax=ax, cmap="coolwarm", annot=True, 
                vmin=wgt_mat.min(), vmax=wgt_mat.max())
    plt.show()
