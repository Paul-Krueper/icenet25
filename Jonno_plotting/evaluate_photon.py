import xgboost
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import os
import re

import argparse

from utils import *
# #EEm #python evaluate.py -i /vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_test_EEm.parquet -d  /vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_test_EEm.parquet   
# --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEm.yml/modeltag__repeatbest_butwithrightiso/beta_0.02__sigma_0.025__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl
#  --m1-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEm.yml/modeltag__repeatbest_butwithrightiso/INIT/ICEBOOST3D/ICEBOOST3D_549.pkl
#   --zscore-file /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEm.yml/modeltag__repeatbest_butwithrightiso/beta_0.02__sigma_0.025__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/zscore.pkl

# #EEp python evaluate.py -i /vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_test_EEp.parquet -d  /vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_test_EEp.parquet 
#  --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEp.yml/modeltag__repeatbest_butwithrightiso/beta_0.05__sigma_0.1__lr_0.05__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl 
#  --m1-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEp.yml/modeltag__repeatbest_butwithrightiso/INIT/ICEBOOST3D/ICEBOOST3D_549.pkl
#   --zscore-file  /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEp.yml/modeltag__repeatbest_butwithrightiso/beta_0.05__sigma_0.1__lr_0.05__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/zscore.pkl

# #EB python evaluate.py -i /vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_test_EB.parquet -d  /vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_test_EB.parquet  
# --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EB.yml/modeltag__repeatbest_butwithrightiso/beta_0.02__sigma_0.04375__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl 
# --m1-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EB.yml/modeltag__repeatbest_butwithrightiso/INIT/ICEBOOST3D/ICEBOOST3D_549.pkl
#  --zscore-file /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EB.yml/modeltag__repeatbest_butwithrightiso/beta_0.02__sigma_0.04375__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/zscore.pkl



#previously:
#EEm #python evaluate.py -i /vols/cms/pfk18/icenet_files/processed/MC_test_EEm.parquet -d  /vols/cms/pfk18/icenet_files/processed/Data_test_EEm.parquet  
#  --m1-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint_bestEEm/zee/config__tune0_EEm.yml/modeltag__GRIDTUNE/INIT/ICEBOOST3D/ICEBOOST3D_549.pkl 
#  --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEm.yml/modeltag__GRIDTUNE/withinti_Mixedw_beta_0.02__sigma_0.025__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl 
#  --zscore-file /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEm.yml/modeltag__GRIDTUNE/withinti_Mixedw_beta_0.02__sigma_0.025__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/zscore.pkl 
# 
#EEp python evaluate.py -i /vols/cms/pfk18/icenet_files/processed/MC_test_EEp.parquet -d  /vols/cms/pfk18/icenet_files/processed/Data_test_EEp.parquet 
#   --m1-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EEp.yml/modeltag__test_EEp-swdbeta0.05-noisereg0.1/test_EEp-swdbeta0.05-noisereg0.1/ICEBOOST3D/ICEBOOST3D_549.pkl 
#    --m2-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EEp.yml/modeltag__test_EEp-swdbeta0.05-noisereg0.1/test_EEp-swdbeta0.05-noisereg0.1/ICEBOOST-SWD/ICEBOOST-SWD_354.pkl 
#     --zscore-file /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EEp.yml/modeltag__test_EEp-swdbeta0.05-noisereg0.1/test_EEp-swdbeta0.05-noisereg0.1/zscore.pkl
# 
#EB python evaluate.py -i /vols/cms/pfk18/icenet_files/processed/MC_test_EB.parquet -d  /vols/cms/pfk18/icenet_files/processed/Data_test_EB.parquet 
#   --m1-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EB.yml/modeltag__test_EB-swdbeta0.175-noisereg0.04375/test_EB-swdbeta0.175-noisereg0.04375/ICEBOOST3D/ICEBOOST3D_549.pkl 
#    --m2-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EB.yml/modeltag__test_EB-swdbeta0.175-noisereg0.04375/test_EB-swdbeta0.175-noisereg0.04375/ICEBOOST-SWD/ICEBOOST-SWD_534.pkl 
#    --zscore-file /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EB.yml/modeltag__test_EB-swdbeta0.175-noisereg0.04375/test_EB-swdbeta0.175-noisereg0.04375/zscore.pkl


#take m1 and zscore from before (trained on + and -) and take m2 (trained on + only):

#EEm #python evaluate.py -i /vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_test_EEm.parquet -d  /vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_test_EEm.parquet   
# --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEm.yml/modeltag__repeatbest_butwithrightiso/beta_0.02__sigma_0.025__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl
#--m1-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint_bestEEm_wrongiso/zee/config__tune0_EEm.yml/modeltag__GRIDTUNE/INIT/ICEBOOST3D/ICEBOOST3D_549.pkl 
#--zscore-file /vols/cms/pfk18/Mikael2/icenet_new/checkpoint_bestEEm_allSWDvars_wrongiso/zee/config__tune0_EEm.yml/modeltag__GRIDTUNE/withinti_Mixedw_beta_0.02__sigma_0.025__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/zscore.pkl 

#EEp python evaluate.py -i /vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_test_EEp.parquet -d  /vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_test_EEp.parquet 
#  --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EEp.yml/modeltag__repeatbest_butwithrightiso/beta_0.05__sigma_0.1__lr_0.05__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl 
# --m1-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EEp.yml/modeltag__test_EEp-swdbeta0.05-noisereg0.1/test_EEp-swdbeta0.05-noisereg0.1/ICEBOOST3D/ICEBOOST3D_549.pkl 
# --zscore-file /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EEp.yml/modeltag__test_EEp-swdbeta0.05-noisereg0.1/test_EEp-swdbeta0.05-noisereg0.1/zscore.pkl

#EB python evaluate.py -i /vols/cms/pfk18/Jons_5Apr24_Zmmy_full_DY_postEE_renamed.parquet -d Jons_5Apr24_Zmmy_full_Data_EFG.parquet  /vols/cms/pfk18/
# --m2-path /vols/cms/pfk18/Mikael2/icenet_new/checkpoint/zee/config__tune0_EB.yml/modeltag__repeatbest_butwithrightiso/beta_0.02__sigma_0.04375__lr_0.1__gamma_1.5__maxdepth_13__lambda_2.0__alpha_0.05/ICEBOOST-SWD/ICEBOOST-SWD_549.pkl 
#   --m1-path /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EB.yml/modeltag__test_EB-swdbeta0.175-noisereg0.04375/test_EB-swdbeta0.175-noisereg0.04375/ICEBOOST3D/ICEBOOST3D_549.pkl 
#    --zscore-file /vols/cms/pfk18/Mikael2/icenet/checkpoint2/zee/config__tune0_EB.yml/modeltag__test_EB-swdbeta0.175-noisereg0.04375/test_EB-swdbeta0.175-noisereg0.04375/zscore.pkl

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-mc', dest='input_mc', default='dy.parquet', help='Input dy to evaluate model on')
parser.add_argument('-d', '--input-data', dest='input_data', default='data.parquet', help='Input data')
parser.add_argument('--zscore-file', dest='zscore_file', default='zscore.pkl', help='Path to z-score file')
parser.add_argument('--m1-path', dest='m1_path', default='.', help='Path to Model 1 files')
parser.add_argument('--m2-path', dest='m2_path', default='.', help='Path to Model 2 files')
parser.add_argument('-p','--parameters', dest="parameters", default="tau1=1,tau2=1,maxW=100", help="Evaluation parameter values")
parser.add_argument("-o","--out", default="samples_processed")
parser.add_argument("--detector")
args = parser.parse_args()

# Hyperparameters
params = {}
for p in args.parameters.split(","):
    params[p.split("=")[0]] = float(p.split("=")[1])

# Load events to evaluate
print(f" --> Reading {args.input_mc}")
df = pd.read_parquet(args.input_mc, engine="pyarrow")
if args.detector=="EEm":
    events=df[df['photon_ScEta'] < -1.566]
elif args.detector=="EEp":
    events=df[df['photon_ScEta'] > 1.566]
elif args.detector=="EB":
    events=df[(df['photon_ScEta'] > -1.444)&(df['photon_ScEta'] < 1.444)]

for i in events.columns:
    print(i)
events["photon_sieie_corr"]=events["photon_sieie"]
events["photon_sieie"]=events["photon_raw_sieie"]

events["photon_sieip_corr"]=events["photon_sieip"]
events["photon_sieip"]=events["photon_raw_sieip"]

events["photon_s4_corr"]=events["photon_s4"]
events["photon_s4"]=events["photon_raw_s4"]

events["photon_r9_corr"]=events["photon_r9"]
events["photon_r9"]=events["photon_raw_r9"]
        
events["photon_phiWidth_corr"]=events["photon_phiWidth"]
events["photon_phiWidth"]=events["photon_raw_phiWidth"]

events["photon_etaWidth_corr"]=events["photon_etaWidth"]
events["photon_etaWidth"]=events["photon_raw_etaWidth"]
    
events["photon_pfChargedIso_corr"]=events["photon_pfChargedIso"]
events["photon_pfChargedIso"]=events["photon_raw_pfChargedIso"]
        

events["photon_pfChargedIsoWorstVtx_corr"]=events["photon_pfChargedIsoWorstVtx"]
events["photon_pfChargedIsoWorstVtx"]=events["photon_raw_pfChargedIsoWorstVtx"]
        
events["photon_trkSumPtHollowConeDR03_corr"]=events["photon_trkSumPtHollowConeDR03"]
events["photon_trkSumPtHollowConeDR03"]=events["photon_raw_trkSumPtHollowConeDR03"]

events["photon_trkSumPtSolidConeDR04_corr"]=events["photon_trkSumPtSolidConeDR04"]
events["photon_trkSumPtSolidConeDR04"]=events["photon_raw_trkSumPtSolidConeDR04"]
        

events["photon_hcalPFClusterIso_corr"]=events["photon_hcalPFClusterIso"]
events["photon_hcalPFClusterIso"]=events["photon_raw_hcalPFClusterIso"]

events["photon_ecalPFClusterIso_corr"]=events["photon_ecalPFClusterIso"]
events["photon_ecalPFClusterIso"]=events["photon_raw_ecalPFClusterIso"]
        
events["photon_hoe_corr"]=events["photon_hoe"]
events["photon_hoe"]=events["photon_raw_hoe"]

events["photon_mvaID_corr"]=events["photon_mvaID"]
events["photon_mvaID"]=events["photon_mvaID_nano"]
       
        
if "photon_esEnergyOverRawE" in events.columns:
    print("EE DETECTED!")
    events["photon_esEnergyOverRawE_corr"]=events["photon_esEnergyOverRawE"]
    events["photon_esEnergyOverRawE"]=events["photon_raw_esEnergyOverRawE"]

    events["photon_esEffSigmaRR_corr"]=events["photon_esEffSigmaRR"]
    events["photon_esEffSigmaRR"]=events["photon_raw_esEffSigmaRR"]

shape_var_list = ['sieie', 'ecalPFClusterIso', 'trkSumPtHollowConeDR03', 'hcalPFClusterIso', 'pfChargedIso', 'phiWidth', 'trkSumPtSolidConeDR04', 'r9', 'pfChargedIsoWorstVtx', 's4', 'etaWidth', 'mvaID', 'sieip',"eta","pt","esEnergyOverRawE","esEffSigmaRR"]
for i in shape_var_list:
    events[f"probe_{i}"]=events[f"photon_{i}"]
    if i not in ["eta","pt"]:
        events[f"probe_{i}_corr"]=events[f"photon_{i}_corr"]

events["fixedGridRhoAll"]=events["Rho_fixedGridRhoAll"]
events["weight"]=events["weight_central"]*events["genWeight"]



print("rename complete")

for i in events.columns:
    print(i)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model 2
print(f" --> Evaluating model 2")
with open(args.m2_path, "rb") as fpkl:
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
events['rwgt_m2'] = np.clip(rw_transform_with_logits(logits2), 0., 100.)

print(params["tau1"])
print(params["tau2"])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data and add normalised weight columns
print(f" --> Reading {args.input_data}")
df_data = pd.read_parquet(args.input_data, engine="pyarrow")
if args.detector=="EEm":
    events_data=df_data[df_data['photon_ScEta'] < -1.566]
elif args.detector=="EEp":
    events_data=df_data[df_data['photon_ScEta'] > 1.566]
elif args.detector=="EB":
    events_data=df_data[(df_data['photon_ScEta'] > -1.444)&(df_data['photon_ScEta'] < 1.444)]
events_data['weight'] = 1
for i in shape_var_list:
    events_data[f"probe_{i}"]=events_data[f"photon_{i}"]
events_data["fixedGridRhoAll"]=events_data["Rho_fixedGridRhoAll"]

# Add normalised weight for pre S1 plotting
events['weight_norm'] = events['weight']*(events_data['weight'].sum()/events['weight'].sum())
# Normalise MC yield to data after S1 and apply S2 weight
events['w_post_S1'] = (events['weight']*events['rwgt_m1'])*(events_data['weight'].sum()/(events['weight']*events['rwgt_m1']).sum())

events['w_post_S2_unnorm'] = events['w_post_S1']*events['rwgt_m2']
events['w_post_S2'] = events['w_post_S2_unnorm']*(events['w_post_S1'].sum()/events['w_post_S2_unnorm'].sum())
print("norm changed by" +str(events['w_post_S2_unnorm'].sum()/events['w_post_S2'].sum()))
print("norm changed by" +str(events['w_post_S2'].sum()/events['w_post_S1'].sum()))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Concatenate dataframes and save
print(" --> Saving processed dataframes")
if not os.path.isdir(f"{args.out}"):
    os.system(f"mkdir -p {args.out}")

events['y'] = 0

events.to_parquet(f"{args.out}/MC_Zmmy_{args.detector}_processed.parquet", engine="pyarrow")

events_data['y'] = 1

events_data.to_parquet(f"{args.out}/Data_Zmmy_{args.detector}_processed.parquet", engine="pyarrow")



