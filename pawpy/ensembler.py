
# <<<<<<<<<<<<<<<<<<<<<<<<< #
# Module for ensemble model #
# >>>>>>>>>>>>>>>>>>>>>>>>> #

import os
import pandas as pd
import lightgbm as lgb

def gen_ensemble_data(model_dir, train_df, valid_df, 
                      y_var="pawpularity", x_vars=None,
                      norm_preds=100, add_lgb_preds=False, 
                      lgb_params=None, extra_lgb_params=None):
    """
    """
    
    if x_vars is None:
        x_vars = [c for c in train_df.columns if c not in ["id", y_var]]
    pred_cols = []
    
    for nn_dir in os.listdir(model_dir):
    
        pred_id = nn_dir[-11:]
        nn_pred_col = "pred_%s_nn"%(pred_id)
        pred_cols.append(nn_pred_col) 

        train_df = train_df.merge(
            pd.read_csv(os.path.join(model_dir, nn_dir, "train_preds.csv")
                        ).rename(columns={"pred": nn_pred_col}),
            on="id"
        )

        valid_df = valid_df.merge(
            pd.read_csv(os.path.join(model_dir, nn_dir, "valid_preds.csv")
                        ).rename(columns={"pred": nn_pred_col}),
            on="id"
        )
        
        if add_lgb_preds:        
            lgb_pred_col = "pred_%s_lgb"%(pred_id)
            pred_cols.append(lgb_pred_col) 
            
            lgb_train = lgb.Dataset(train_df[x_vars + [nn_pred_col]], 
                                    label=train_df[y_var])
            
            lgb_valid = lgb.Dataset(valid_df[x_vars + [nn_pred_col]], 
                                    label=valid_df[y_var])
            
            lgb_model = lgb.train(
                lgb_params, 
                train_set=lgb_train, 
                valid_sets=lgb_valid,
                num_boost_round=extra_lgb_params["n_rounds"], 
                early_stopping_rounds=extra_lgb_params["early_stop"], 
                verbose_eval=extra_lgb_params["verbose"]
            )
            
            train_df.loc[:, lgb_pred_col] = lgb_model.predict(
                train_df[lgb_model.feature_name()])
            
            valid_df.loc[:, lgb_pred_col] = lgb_model.predict(
                valid_df[lgb_model.feature_name()])
    
    train_df.loc[:, pred_cols] = train_df.loc[:, pred_cols]/norm_preds
    valid_df.loc[:, pred_cols] = valid_df.loc[:, pred_cols]/norm_preds
            
    return train_df, valid_df, x_vars, pred_cols
