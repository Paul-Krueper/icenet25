import xgboost
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import os
import re

import argparse

from utils import *
#EB python evaluate_bootstrap.py -i /vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_test_EB.parquet -d  /vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_test_EB.parquet  --m1-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EB.yml/modeltag__test_EB-swdbeta0.175-noisereg0.04375/test_EB-swdbeta0.175-noisereg0.04375/ICEBOOST3D/ICEBOOST3D_549.pkl --zscore-file /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EB.yml/modeltag__repeatbest_butwithrightiso_bootstrap_posonly/beta_0.02__sigma_0.04375__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/zscore.pkl --m2-path checkpoint/zee/config__tune0_EB.yml/modeltag__repeatbest_butwithrightiso_bootstrap_posonly/beta_0.02__sigma_0.04375__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-mc', dest='input_mc', default='dy.parquet', help='Input dy to evaluate model on')
parser.add_argument('-d', '--input-data', dest='input_data', default='data.parquet', help='Input data')
parser.add_argument('--zscore-file', dest='zscore_file', default='zscore.pkl', help='Path to z-score file')
parser.add_argument('--m1-path', dest='m1_path', default='.', help='Path to Model 1 files')
parser.add_argument('--m2-path', dest='m2_path', default='.', help='Path to Model 2 files')
parser.add_argument('-p','--parameters', dest="parameters", default="tau1=1,tau2=1,maxW=100", help="Evaluation parameter values")
args = parser.parse_args()

# Hyperparameters
params = {}
for p in args.parameters.split(","):
    params[p.split("=")[0]] = float(p.split("=")[1])

# Load events to evaluate
print(f" --> Reading {args.input_mc}")
events = pd.read_parquet(args.input_mc, engine="pyarrow")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model 1 (3D)
print(f" --> Evaluating model 1")
with open(args.m1_path, "rb") as fpkl:
    m1 = pkl.load(fpkl)

# Find best model
best_idx = int(np.argmin(m1['losses']['val_losses']))
best_model = m1['model'][:best_idx+1] 
print(f"  * Using model at epoch: {best_idx}")

# No Z-score normalisation for model 1
input_features = best_model.feature_names
X3D = xgboost.DMatrix(events[input_features])
# Evaluate model
logits1 = best_model.predict(X3D)
logits1 /= params['tau1']
events['rwgt_m1'] = np.clip(rw_transform_with_logits(logits1), 0., 100.)

model_list_ext=[""]
for nu in range(1,11):
    model_list_ext.append(f"__bs_{nu}")

for i in model_list_ext:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model 2
    print(f" --> Evaluating model 2")
    with open(f"../{args.m2_path}/ICEBOOST-SWD{i}/ICEBOOST-SWD{i}_549.pkl", "rb") as fpkl:
        m2 = pkl.load(fpkl)

    # Find best model
    best_idx = int(np.argmin(m2['losses']['val_losses']))
    best_model = m2['model'][:best_idx+1]
    print(f"  * Using model at epoch: {best_idx}")

    # Apply z-score normalisation for model 2
    with open(args.zscore_file, "rb") as fpkl:
        zscore = pkl.load(fpkl)
    input_features = zscore['ids']
    X = (events[input_features].to_numpy()-zscore['X_mu'])/zscore['X_std']
    X = xgboost.DMatrix(X, feature_names=input_features)
    # Evaluate modoel
    logits2 = best_model.predict(X)
    logits2 /= params['tau2']
    events[f'rwgt_m2{i}'] = np.clip(rw_transform_with_logits(logits2), 0., 100.)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data and add normalised weight columns
    print(f" --> Reading {args.input_data}")
    events_data = pd.read_parquet(args.input_data, engine="pyarrow")
    events_data['weight'] = 1

    # Add normalised weight for pre S1 plotting
    events['weight_norm'] = events['weight']*(events_data['weight'].sum()/events['weight'].sum())

    # Normalise MC yield to data after S1 and apply S2 weight
    events['w_post_S1'] = (events['weight']*events['rwgt_m1'])*(events_data['weight'].sum()/(events['weight']*events['rwgt_m1']).sum())
    events[f'w_post_S2_unnorm{i}'] = events['w_post_S1']*events[f'rwgt_m2{i}']
    events[f'w_post_S2{i}'] = events[f'w_post_S2_unnorm{i}']*(events['w_post_S1'].sum()/events[f'w_post_S2_unnorm{i}'].sum())

bootstrap_columns_df=events[[ "w_post_S2"+x for x in model_list_ext]]
events["bs_std"]=bootstrap_columns_df.std()
print(events["bs_std"])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Concatenate dataframes and save
print(" --> Saving processed dataframes")
if not os.path.isdir("samples_processed"):
    os.system("mkdir -p samples_processed")

events['y'] = 0
name = re.sub(".parquet", "_processed_boo.parquet", args.input_mc.split("/")[-1])
events.to_parquet(f"samples_processed/{name}", engine="pyarrow")

events_data['y'] = 1
name = re.sub(".parquet", "_processed_boo.parquet", args.input_data.split("/")[-1])
events_data.to_parquet(f"samples_processed/{name}", engine="pyarrow")



