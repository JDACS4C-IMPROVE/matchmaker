""" Preprocess original data to generate datasets for the
Matchmaker prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["ml_data_outdir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].

2. Y data files are not generated seperately for the original data
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specific imports
import os
import MatchMaker
import pickle

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Note! This list should not be modified (i.e., no params should added or
# removed from the list.
# 
# There are two types of params in the list: default and required
# default:   default values should be used
# required:  these params must be specified for the model in the param file
app_preproc_params = []

# 2. Model-specific params 
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
    {"name": "comb-data-name",
     "type": str,
     "default": 'data/DrugCombinationData.tsv',
     "help": "Name of the drug combination data",
    },
    {"name": "cell_line-gex",
     "type": str,
     "default": 'data/cell_line_gex.csv',
     "help": "Name of the cell line gene expression data",
    },
    {"name": "drug1-chemicals",
     "type": str,
     "default": "data/drug1_chem.csv",
     "help": "Name of the chemical features data for drug 1",
    },
    {"name": "drug2-chemicals",
     "type": str,
     "default": "data/drug2_chem.csv",
     "help": "Name of the chemical features data for drug 2",
    },
    {"name": "gpu-devices",
     "type": str,
     "default": "0",
     "help": "gpu device ids for CUDA_VISIBLE_DEVICES",
    },
    {"name": "train-ind",
     "type": str,
     "default": "data/train_inds.txt",
     "help": "Data indices that will be used for training",
    },
    {"name": "val-ind",
     "type": str,
     "default": "data/val_inds.txt",
     "help": "Data indices that will be used for validation",
    },
    {"name": "test-ind",
     "type": str,
     "default": "data/test_inds.txt",
     "help": "Data indices that will be used for test",
    },
    {"name": "arch",
     "type": str,
     "default": "matchmaker/architecture.txt",
     "help": "Architecute file to construct MatchMaker layers",
    },
    {"name": "gpu-support",
     "type": frm.str2bool,
     "default": True,
     "help": "Use GPU support or not",
    },
    {"name": "l_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate",
    },
    {"name": "inDrop",
     "type": float,
     "default": 0.0001,
     "help": "inDrop",
    },
    {"name": "drop",
     "type": float,
     "default": 0.5,
     "help": "drop",
    },
    {"name": "num_cores",
     "type": int,
     "default": 8,
     "help": "num_cores",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------


# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    ##params = frm.build_paths(params)  

    # Create output dir for model input data (to save preprocessed ML data)
    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Use the provided data loaders to load data that is required by the model.
    #
    # Benchmark data includes three dirs: x_data, y_data, splits.
    # The x_data contains files that represent feature information such as
    # cancer representation (e.g., omics) and drug representation (e.g., SMILES).
    #
    # Prediction models utilize various types of feature representations.
    # Drug response prediction (DRP) models generally use omics and drug features.
    #
    # If the model uses omics data types that are provided as part of the benchmark
    # data, then the model must use the provided data loaders to load the data files
    # from the x_data dir.
   # load and process data
    chem1, chem2, cell_line, synergies = MatchMaker.data_loader(params["drug1_chemicals"], params["drug2_chemicals"],
                                                params["cell_line_gex"], params["comb_data_name"])
    # normalize and split data into train, validation and test
    norm = 'tanh_norm'
    train_data, val_data, test_data = MatchMaker.prepare_data(chem1, chem2, cell_line, synergies, norm,
                                            params["train_ind"], params["val_ind"], params["test_ind"])

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.

    with open("train_data.pkl", 'wb+') as f:
        pickle.dump(train_data, f)

    with open("val_data.pkl", 'wb+') as f:
        pickle.dump(val_data, f)
    
    with open("test_data.pkl", 'wb+') as f:
        pickle.dump(test_data, f)
   

    return params["ml_data_outdir"]


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params_original.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])