""" Inference with Matchmaker for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["infer_outdir"].

1. Predictions on test data.
   Raw model predictions calcualted using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# Model-specific imports
import os
import numpy as np
import tensorflow as tf
import MatchMaker
import pickle
import tensorflow.keras as keras

# [Req] Imports from preprocess and train scripts
from matchmaker_preprocess_improve import preprocess_params
from matchmaker_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# ---------------------


# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")


    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    with open("test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
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
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]
    ##########################
    # load the best model
    #model.load_weights(modelName)
    model = keras.models.load_model(str(modelpath))
    # predict in Drug1, Drug2 order
    pred1 = MatchMaker.predict(model, [test_data['drug1'],test_data['drug2']])
    # predict in Drug2, Drug1 order
    pred2 = MatchMaker.predict(model, [test_data['drug2'],test_data['drug1']])
    # take the mean for final prediction
    test_pred = (pred1 + pred2) / 2
    test_true = test_data['y']
    ############ from model


    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params_original.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_args,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])