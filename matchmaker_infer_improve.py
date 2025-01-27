""" Inference with Matchmaker for drug response prediction.
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from model_params_def import infer_params

# Model-specific imports
import os
import numpy as np
import tensorflow as tf
import MatchMaker
import pickle
import tensorflow.keras as keras


filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params):
     # --------------------------------------------------------------------
    # [Req] Create data names for test set and build model path
    # --------------------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])

    test_data_path = params["input_data_dir"] + "/" + test_data_fname
    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    with open(test_data_path, 'rb') as f:
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
        y_true=test_true,
        y_pred=test_pred,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )


    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true,
            y_pred=test_pred,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )


    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params_v0.1data.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_args,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])