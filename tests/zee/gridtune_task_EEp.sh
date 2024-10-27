#!/bin/sh
#
# (Modify this file!)
#
# m.mieskolainen@imperial.ac.uk, 2024

SETCONDA="/home/hep/mmieskol/setconda.sh"
ICEPATH="/vols/cms/mmieskol/icenet"

DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
CONFIG="tune0_EEp"
MODELTAG="none"
maxevents=500000

BETA_ARRAY=(0.0 0.0025 0.005 0.01 0.02 0.04)
SIGMA_ARRAY=(0.0 0.025 0.05 0.1 0.2)

#SWD_VAR="['.*']" # all
SWD_VAR="['fixedGridRhoAll', 'probe_eta', 'probe_pt']"

source $ICEPATH/tests/zee/helper.sh
