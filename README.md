# IMPROVE - MatchMaker: Drug Synergy Prediction

---

This is the IMPROVE implementation of the original model with original data.

## Dependencies and Installation
### Conda Environment
```
conda create --name matchmaker_IMPROVE python=3.7 numpy=1.18.1 scipy=1.4.1 pandas=1.0.1 tensorflow-gpu=2.1.0 scikit-learn=0.22.1 keras-metrics=1.1.0 h5py=2.10.0
conda activate matchmaker_IMPROVE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

### Clone this repository
```
git clone https://github.com/JDACS4C-IMPROVE/matchmaker
cd matchmaker
git checkout IMPROVE-original
cd ..
```

### Clone IMPROVE repository
```
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
cd ..
```

### Download Original Data
You can download preprocessed data from <a href="https://drive.google.com/open?id=1eQpwJKiIdMI0wTz_GEa285q0GHUr6wRe">**link**</a>, extract all files into `data/`


## Running the Model
Activate the conda environment:

```
conda activate matchmaker_IMPROVE
```

Set environment variables:
```
export IMPROVE_DATA_DIR="./"
export PYTHONPATH=$PYTHONPATH:/your/path/to/IMPROVE
```

Run preprocess, train, and infer scripts:
```
python matchmaker/matchmaker_preprocess_improve.py
python matchmaker/matchmaker_train_improve.py
python matchmaker/matchmaker_infer_improve.py
```



## References
Original GitHub: https://github.com/tastanlab/matchmaker

Original paper: https://pubmed.ncbi.nlm.nih.gov/34086576/
