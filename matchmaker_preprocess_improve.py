""" Preprocess original data to generate datasets for the
Matchmaker prediction model.

"""

import sys
from pathlib import Path
from typing import Dict
# [Req] Core improvelib imports
from improvelib.applications.synergy.config import SynergyPreprocessConfig
import improvelib.utils as frm
# Model-specific imports
import numpy as np
import pandas as pd
import MatchMaker
import pickle
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]


    #norm = 'tanh_norm' 
    # need to add this norm, using std for now



# [Req]
def run(params: Dict):
    # ------------------------------------------------------
    # [Req] Load feature data
    # ------------------------------------------------------
    print("Load omics data.")
    cell_feature = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])

    print("Load drug data.")
    drug_feature = frm.get_x_data(file = params['drug_mordred_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    print("original column names:", drug_feature.columns)

    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    # not needed for this data/model

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    print("Load train response data.")
    response_train = frm.get_y_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   y_data_file=params['y_data_file'])
    response_train = response_train.dropna(subset=[params['y_col_name']])
    
    print("Find intersection of training data.")
    response_train = frm.get_y_data_with_features(response_train, cell_feature, params['canc_col_name'])
    response_train = frm.get_y_data_with_features(response_train, drug_feature, params['drug_1_col_name'])
    response_train = frm.get_y_data_with_features(response_train, drug_feature, params['drug_2_col_name'])
    omics_train = frm.get_features_in_y_data(cell_feature, response_train, params['canc_col_name'])
    drug1_train = frm.get_features_in_y_data(drug_feature, response_train, params['drug_1_col_name'])
    drug2_train = frm.get_features_in_y_data(drug_feature, response_train, params['drug_2_col_name'])
    drugs_train = pd.concat([drug1_train, drug2_train]).drop_duplicates()
    print("columns drug_1 before det transform:", drug1_train.columns)
    print("columns drug_2 before det transform:", drug2_train.columns)
    print("columns drugs before det transform:", drugs_train.columns)
    print("Determine transformations.")
    frm.determine_transform(omics_train, 'omics_transform', params['cell_transcriptomic_transform'], params['output_dir'])
    frm.determine_transform(drugs_train, 'drugs_transform', params['drug_mordred_transform'], params['output_dir'])


    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = frm.get_y_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                y_data_file=params['y_data_file'])
        response_stage = response_stage.dropna(subset=[params['y_col_name']])
        response_stage = frm.get_y_data_with_features(response_stage, cell_feature, params['canc_col_name'])
        response_stage = frm.get_y_data_with_features(response_stage, drug_feature, params['drug_1_col_name'])
        response_stage = frm.get_y_data_with_features(response_stage, drug_feature, params['drug_2_col_name'])
        omics_stage = frm.get_features_in_y_data(cell_feature, response_stage, params['canc_col_name'])
        drug1_stage = frm.get_features_in_y_data(drug_feature, response_stage, params['drug_1_col_name'])
        drug2_stage = frm.get_features_in_y_data(drug_feature, response_stage, params['drug_2_col_name'])
        drugs_stage = pd.concat([drug1_stage, drug2_stage]).drop_duplicates()
        print("columns drug_1 before transform:", drug1_stage.columns)
        print("columns drug_2 before transform:", drug2_stage.columns)
        print("columns drugs before transform:", drugs_stage.columns)
        print(f"Transform {stage} data.")
        omics_stage = frm.transform_data(omics_stage, 'omics_transform', params['output_dir'])
        drugs_stage = frm.transform_data(drugs_stage, 'drugs_transform', params['output_dir'])

        print(f"Merge {stage} data")
        y_df_cols = response_stage.columns.tolist()
        # prefix drug and cell features
        omics_stage = omics_stage.add_prefix("cell_")
        drug1_stage = drugs_stage.add_prefix("drug1_")
        drug2_stage = drugs_stage.add_prefix("drug2_")
        print("columns drug_1 after add prefix:", drug1_stage.columns)
        print("columns drug_2 after add prefix:", drug2_stage.columns)
        data = response_stage.merge(omics_stage, on=params["canc_col_name"], how="inner")
        data = data.merge(drug1_stage, left_on=params["drug_1_col_name"], right_on=params["drug_col_name"], how="inner")
        data = data.merge(drug2_stage, left_on=params["drug_2_col_name"], right_on=params["drug_col_name"], how="inner")

        print(f"Save {stage} data")
        stage_data = {}
        if stage == 'train':
            # pull out cell and drugs data
            drug_1 = data.loc[:, data.columns.str.startswith('drug1_')]
            drug_1 = drug_1.columns.str.removeprefix('drug1_')
            print("shape drug_1:", drug_1.shape)
            print("columns drug_1:", drug_1.columns)
            print("dtypes drug_1", drug_1.dtypes)
            drug_2 = data.loc[:, data.columns.str.startswith('drug2_')]
            drug_2 = drug_2.columns.str.removeprefix('drug2_')
            print("shape drug_2:", drug_2.shape)
            print("columns drug_2:", drug_2.columns)
            print("dtypes drug_2", drug_2.dtypes)
            cell = data.loc[:, data.columns.str.startswith('cell_')]
            # training data for matchmaker is done twice, drug1-drug2 and drug2-drug1
            drug_first_order = pd.concat([drug_1, drug_2])
            drug_second_order = pd.concat([drug_2, drug_1])
            cell_both_order = pd.concat([cell, cell])
            # concat cell to drug for each
            drug1_cell = pd.concat([drug_first_order, cell_both_order], axis=1)
            drug2_cell = pd.concat([drug_second_order, cell_both_order], axis=1)
            y_alone = pd.concat([data[params['y_col_name']], data[params['y_col_name']]])
            # save to dict
            stage_data['drug1'] = np.array(drug1_cell)
            stage_data['drug2'] = np.array(drug2_cell)
            stage_data['y'] = np.array(y_alone)
        else:
            # pull out cell and drugs data
            drug_1 = data.loc[:, data.columns.str.startswith('drug1_')]
            drug_1 = drug_1.rename(columns = lambda x: x.strip('drug1_'))
            drug_2 = data.loc[:, data.columns.str.startswith('drug2_')]
            drug_2 = drug_2.rename(columns = lambda x: x.strip('drug2_'))
            cell = data.loc[:, data.columns.str.startswith('cell_')]
            # concat cell to drug for each
            drug1_cell = pd.concat([drug_1, cell], axis=1)
            drug2_cell = pd.concat([drug_2, cell], axis=1)
            y_alone = data[params['y_col_name']]
            # save to dict
            stage_data['drug1'] = np.array(drug1_cell)
            stage_data['drug2'] = np.array(drug2_cell)
            stage_data['y'] = np.array(y_alone)
        print(f"___ STAGE {stage} ____")
        print("stage_data['drug1'].shape:", stage_data['drug1'].shape)
        print("stage_data['drug2'].shape:", stage_data['drug2'].shape)
        print("stage_data['y'].shape:", stage_data['y'].shape)
        # Save x data
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        data_path = params["output_dir"] + "/" + data_fname
        with open(data_path, 'wb+') as f:
            pickle.dump(stage_data, f, protocol=4)
        # [Req] Save y dataframe for the current stage
        ydf = data[y_df_cols]
        frm.save_stage_ydf(ydf, stage, params["output_dir"])
    print(f"Training data here was {params['train_split_file']}")
    return params["output_dir"]


# [Req]
def main(args):
    cfg = SynergyPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="matchmaker_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])