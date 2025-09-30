# IMPROVE - MatchMaker: Drug Synergy Prediction
---

This repository demonstrates how to use the [IMPROVE library](https://jdacs4c-improve.github.io/docs/) for building a synergy prediction model using Matchmaker.


## Dependencies
Installation instructions are detailed below in [Step-by-step instructions](#step-by-step-instructions).


ML framework:
+ [TensorFlow](https://www.tensorflow.org/)

IMPROVE dependencies:
+ [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE)

## Dataset
Benchmark data for Synergy can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/synergy_data_v0.2.0).



# Step-by-step instructions

### 1. Clone the model repository and checkout the develop branch (or tag of your choice)
```bash
git clone https://github.com/JDACS4C-IMPROVE/matchmaker
cd matchmaker
git checkout develop
```


### 2. Set computational environment
```bash
conda create --name matchmaker_IMPROVE python=3.7 numpy=1.18.1 scipy=1.4.1 pandas=1.0.1 tensorflow-gpu=2.1.0 scikit-learn=0.22.1 keras-metrics=1.1.0 h5py=2.10.0
conda activate matchmaker_IMPROVE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```


### 3. Preprocess benchmark data to construct model input data 
```bash
python matchmaker_preprocess_improve.py --input_dir ./synergy_data_v0.2.0 --output_dir exp_result
```

Preprocesses the data and creates train, validation (val), and test datasets.

Generates:
* three model input data files
* three tabular data files, each containing the synergy values and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`



### 4. Train model
```bash
python matchmaker_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains a model using the model input data.

Generates:
* trained model
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`


### 5. Run inference on test data with the trained model
```bash
python matchmaker_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`





## References
Original GitHub: https://github.com/tastanlab/matchmaker

Original paper: https://pubmed.ncbi.nlm.nih.gov/34086576/
