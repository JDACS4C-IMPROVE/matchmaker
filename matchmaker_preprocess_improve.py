""" Preprocess original data to generate datasets for the
Matchmaker prediction model.

"""

import sys
from pathlib import Path
from typing import Dict
# [Req] Core improvelib imports
from improvelib.applications.synergy.config import SynergyPreprocessConfig
import improvelib.applications.synergy.synergy_utils as syn
from improvelib.utils import str2bool
import improvelib.utils as frm
# Model-specific imports
import numpy as np
import pandas as pd
import joblib
import os
import MatchMaker
import pickle
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]



# [Req]
def run(params: Dict):
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val/test sets
    # --------------------------------------------------------------------
    data_train_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    data_val_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")
    data_test_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    # Create output dir for model input data (to save preprocessed ML data)
    #frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    print("File reading ...")
 

    # need to make this work for all feature types AND MULTIPLE
    # data file names
    # eventually fix this in improvelib so it checks for a path, and then in the input_dir
    # read in data
    y_data = syn.get_all_response_data(train_split_file = params['train_split_file'], 
                                   val_split_file = params['val_split_file'], 
                                   test_split_file = params['test_split_file'], 
                                   benchmark_dir = params['input_dir'])
    cell_feature = syn.get_cell_transcriptomics(file = params['cell_transcriptomic_file'], 
                                                  benchmark_dir = params['input_dir'], 
                                                  cell_column_name = params['cell_column_name'], 
                                                  norm = params['cell_transcriptomic_transform'])
    drug_feature = syn.get_drug_mordred(file = params['drug_mordred_file'], 
                     benchmark_dir = params['input_dir'], 
                     drug_column_name = params['drug_column_name'])


    # prefix drug and cell features
    cell_feature = cell_feature.add_prefix("cell_")
    drug1_feature = drug_feature.add_prefix("drug1_")
    drug2_feature = drug_feature.add_prefix("drug2_")

    # join all datasets on inner
    y_cell = y_data.join(cell_feature, on=params['cell_column_name'], how="inner")
    y_cell_d1 = y_cell.join(drug1_feature, on=params['drug_1_column_name'], how="inner")
    y_cell_d1_d2 = y_cell_d1.join(drug2_feature, on=params['drug_2_column_name'], how="inner")
    y_cell_d1_d2 = y_cell_d1_d2.dropna(subset=[params["y_col_name"]])
    y_cell_d1_d2 = y_cell_d1_d2.reset_index(drop=True)

    # pull out features
    drug2_indexed = y_cell_d1_d2.loc[:, y_cell_d1_d2.columns.str.startswith('drug2_')]
    drug1_indexed = y_cell_d1_d2.loc[:, y_cell_d1_d2.columns.str.startswith('drug1_')]
    cell_indexed = y_cell_d1_d2.loc[:, y_cell_d1_d2.columns.str.startswith('cell_')]

    # drop features for y data
    y_indexed = y_cell_d1_d2.drop(y_cell_d1_d2.columns[y_cell_d1_d2.columns.str.startswith(("drug2_", "drug1_", "cell_"))], axis=1)

    # reindexed splits
    train_index = y_indexed.index[y_indexed['split'] == "train"].to_frame()
    val_index = y_indexed.index[y_indexed['split'] == "val"].to_frame()
    test_index = y_indexed.index[y_indexed['split'] == "test"].to_frame()

    train_sp = params["output_dir"] + "/train_split.txt"
    val_sp = params["output_dir"] + "/val_split.txt"
    test_sp = params["output_dir"] + "/test_split.txt"

    train_index.to_csv(train_sp, index=False, header=False)
    val_index.to_csv(val_sp, index=False, header=False)
    test_index.to_csv(test_sp, index=False, header=False)
    
    # np for prepare_data()
    cell_line = np.array(cell_indexed.values)
    chem1 = np.array(drug1_indexed.values)
    chem2 = np.array(drug2_indexed.values)
    synergies = np.array(y_indexed[params["y_col_name"]])

    print("Files read.")
    print("File preparing ...")
    # normalize and split data into train, validation and test
    norm = 'tanh_norm'


    train_data, val_data, test_data = MatchMaker.prepare_data(chem1, chem2, cell_line, synergies, norm,
                                            train_sp, val_sp, test_sp)
    print("Files prepared.")
    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    train_data_path = params["output_dir"] + "/" + data_train_fname
    val_data_path = params["output_dir"] + "/" + data_val_fname
    test_data_path = params["output_dir"] + "/" + data_test_fname

    with open(train_data_path, 'wb+') as f:
        pickle.dump(train_data, f, protocol=4)

    with open(val_data_path, 'wb+') as f:
        pickle.dump(val_data, f, protocol=4)
    
    with open(test_data_path, 'wb+') as f:
        pickle.dump(test_data, f, protocol=4)

    # --------------------------------------------------------------------
    # [Req] Save response data (Y data)
    # --------------------------------------------------------------------
    ydf_train = y_indexed[y_indexed["split"] == "train"]
    ydf_val = y_indexed[y_indexed["split"] == "val"]
    ydf_test = y_indexed[y_indexed["split"] == "test"]
    frm.save_stage_ydf(ydf=ydf_train, stage="train", output_dir=params["output_dir"])
    frm.save_stage_ydf(ydf=ydf_val, stage="val", output_dir=params["output_dir"])
    frm.save_stage_ydf(ydf=ydf_test, stage="test", output_dir=params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    cfg = SynergyPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="matchmaker_params.ini",
        additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])