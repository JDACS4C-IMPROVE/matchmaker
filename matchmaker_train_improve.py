""" Train Matchmaker for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["model_outdir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].

2. Predictions on val data. 
   Raw model predictions calcualted using the trained model on val data. The
   predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in val_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# Model-specific imports
import os
import tensorflow as tf
import MatchMaker
import pickle

# [Req] Imports from preprocess script
from matchmaker_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = []

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + model_train_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req]
def run(params):
    """ Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """



# save model

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    #frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    #train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    #val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]
    # read data from preprocess
    with open("train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)

    with open("val_data.pkl", 'rb') as f:
        val_data = pickle.load(f)
    

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_devices"]
    if params["gpu_support"]:
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 2
        num_GPU = 0

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=params["num_cores"],
                            inter_op_parallelism_threads=params["num_cores"],
                            allow_soft_placement=True,
                            device_count = {'CPU' : num_CPU,
                                            'GPU' : num_GPU}
                        )

    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    # calculate weights for weighted MSE loss

    min_s = np.amin(train_data['y'])
    loss_weight = np.log(train_data['y'] - min_s + np.e)

    # load architecture file
    architecture = pd.read_csv(params["arch"])

    # prepare layers of the model and the model name
    layers = {}
    layers['DSN_1'] = architecture['DSN_1'][0] # layers of Drug Synergy Network 1
    layers['DSN_2'] = architecture['DSN_2'][0] # layers of Drug Synergy Network 2
    layers['SPN'] = architecture['SPN'][0] # layers of Synergy Prediction Network

    model = MatchMaker.generate_network(train_data, layers, params["inDrop"], params["drop"])

    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------
    model = MatchMaker.trainer(model, params["l_rate"], train_data, val_data, params["max_epoch"], params["this_batch_size"],
                                params["earlyStop_patience"], params["model_name"], loss_weight)

    # -----------------------------
    # Save model
    # -----------------------------
    model.save(str(modelpath))

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Compute predictions
    val_true = val_data['y']
    # predict in Drug1, Drug2 order
    pred1 = MatchMaker.predict(model, [val_data['drug1'],val_data['drug2']])
    # predict in Drug2, Drug1 order
    pred2 = MatchMaker.predict(model, [val_data['drug2'],val_data['drug1']])
    # take the mean for final prediction
    val_pred = (pred1 + pred2) / 2

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores


def initialize_parameters():
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params_original.txt",
        additional_definitions=additional_definitions,
        # required=req_train_args,
        required=None,
    )
    return params


# [Req]
def main(args):
# [Req]
    params = initialize_parameters()
    val_scores = run(params)
    print("\nFinished training GraphDRP model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])