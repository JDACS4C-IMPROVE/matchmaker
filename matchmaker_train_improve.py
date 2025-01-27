""" Train Matchmaker for drug response prediction.
"""
import sys
from pathlib import Path
from typing import Dict
# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics
from model_params_def import train_params
# Model-specific imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import MatchMaker
import pickle
import tensorflow.keras as keras
from model_params_def import train_params

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params):
# save model
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])
    

    train_data_path = params["input_dir"] + "/" + train_data_fname
    val_data_path = params["input_dir"] + "/" + val_data_fname
    

    # read data from preprocess
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(val_data_path, 'rb') as f:
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
    model = MatchMaker.trainer(model, params["learning_rate"], train_data, val_data, params["epochs"], params["batch_size"],
                                params["patience"], params["model_file_name"], loss_weight)

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
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )


    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores



# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="params_v0.1data.txt",
        additional_definitions=train_params)
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])