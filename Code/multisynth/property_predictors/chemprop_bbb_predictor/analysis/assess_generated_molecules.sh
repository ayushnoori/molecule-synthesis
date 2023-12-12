#!/bin/bash

for NAME in chemprop_rdkit_bbb_mcts random
do
python SyntheMol/inaki_assess_generated_molecules.py \
    --data_path Data/generations/${NAME}/molecules.csv \
    --save_dir Data/generations/${NAME} \
    --reference_paths Data/property_predictors/chemprop_bbb_predictor/preprocessed_data.csv
done