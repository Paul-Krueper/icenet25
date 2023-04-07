#!/bin/sh
#
# Execute distributed deployment for the DQCD analysis
# Run with: source runme.sh

# Set system memory limits
ulimit -s unlimited # stack
#ulimit -v unlimited # virtual memory

CONFIG="tune0.yml"
#DATAPATH="/home/user/travis-stash/input/icedqcd"
DATAPATH="/vols/cms/mc3909"

# Grid (distributed) processing (set per node via export GRID_ID=0; export GRID_NODES=10)
#GRID_ID=0
#GRID_NODES=1

# This needs to be executed only once
#python configs/dqcd/include/ymlgen.py --process 'vector' --filerange '[6-100000]' --outputfile configs/dqcd/include/vector_deploy.yml

# Use * or other glob wildcards for filenames
python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/vector_deploy.yml' --modeltag vector_all --grid_id $GRID_ID --grid_nodes $GRID_NODES --config $CONFIG --datapath $DATAPATH
