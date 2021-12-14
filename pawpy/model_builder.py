
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Module for building transfer learning framework #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kreg
import tensorflow.keras.initializers as kini
    
def build_NN_model(config, model_map):
    """
    """
    
    # load config variables
    model_type = config["model_type"]
    img_dim = config["img_dim"] 
    params = config["params"]
    
    # load pre-trained model
    pre_net = model_map[model_type]["model"](
        weights='imagenet', 
        input_shape=(img_dim, img_dim, 3), 
        include_top=False
    )
    pre_net.trainable = False
    
    # init input list for neural network
    all_inputs = []
  
    # image input    
    img_in = kl.Input(shape=(img_dim, img_dim, 3))
    all_inputs.append(img_in)

    # pre-processing
    if model_map[model_type]["preproc"] is not None:
        img_x = model_map[model_type]["preproc"](img_in)
        img_x = pre_net(img_x, training=False)
    else:
        img_x = pre_net(img_in, training=False)

    # tabular metadata inputs
    x_in = kl.Input(shape=len(config["tab_feats"]))
    all_inputs.append(x_in)    

    # image data processing
    if config["conv_proc"]:
        d = model_map[model_type]["final_shape"]
        all_x = kl.Reshape((d[0], d[1]*d[2], 1))(img_x)
        all_x = kl.Conv2D(
            filters=params["conv"]["nf"], 
            kernel_size=d[:2], strides=d[:2], name="post_conv2D",
            kernel_regularizer=kreg.l2(params["conv"]["l2"]),
            kernel_initializer=kini.RandomUniform(
                minval=1/((d[0]+1)*(d[1]+1)),
                maxval=1/((d[0]-1)*(d[1]-1))
            )
        )(all_x)
        all_x = kl.Flatten()(all_x)
    else:
        all_x = kl.GlobalAvgPool2D()(img_x)

    # add tabular features and then dropout
    if config["batch_norm"]:
        all_x = kl.BatchNormalization()(all_x)    
    all_x = kl.Concatenate()([all_x, x_in])
    all_x = kl.Dropout(params["drop"])(all_x)

    # additional dense layer
    if config["extra_dense"]:
        all_x = kl.Dense(
            params["xtra"]["n"], activation="relu", 
            kernel_regularizer=kreg.l2(params["xtra"]["l2"]), 
            name="extra_dense"
        )(all_x)

    # final output layer
    all_x = kl.Dense(
        1, activation="sigmoid", name="final_layer", 
        kernel_regularizer=kreg.l2(params["outy"]["l2"])
    )(all_x)
    out_y = kl.Lambda(lambda x: x * 100)(all_x)

    # compile model
    model = tf.keras.Model(inputs=all_inputs, outputs=out_y)
    return(model)
