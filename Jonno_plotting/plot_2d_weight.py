import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import os
import re

from utils import *

mplhep.style.use("CMS") 
mplhep.style.use({"savefig.bbox": "tight"})

kin_var_list = ['probe_pt', 'probe_eta', 'fixedGridRhoAll']
shape_var_list = ['probe_sieie', 'probe_ecalPFClusterIso', 'probe_trkSumPtHollowConeDR03', 'probe_hcalPFClusterIso', 'probe_pfChargedIso', 'probe_phiWidth', 'probe_trkSumPtSolidConeDR04', 'probe_r9', 'probe_pfChargedIsoWorstVtx', 'probe_s4', 'probe_etaWidth', 'probe_mvaID', 'probe_sieip']
detector_name={"EB":"EB","EEp":"EE+","EEm":"EE-"}
percentiles = (16,80)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--xvar', dest='xvar', default='corrFactorS2', help='Variable on x-axis')
parser.add_argument('--xlim', dest='xlim', default="percentile", type=str, help='x-axis range')
parser.add_argument('--ylim', dest='ylim', default="percentile", type=str, help='y-axis range')
parser.add_argument('--yvar', dest='yvar', default='probe_eta', help='Variable on y-axis')
parser.add_argument('--nbinsx', dest='nbinsx', default=20, type=int, help='Number of bins in x-axis')
parser.add_argument('--nbinsy', dest='nbinsy', default=20, type=int, help='Number of bins in y-axis')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--plot-path', dest='plot_path', default="plots", help='Path to save plots')
parser.add_argument("-i","--inp",default="samples_processed")
args = parser.parse_args()
if not os.path.isdir(args.plot_path):
        os.system(f"mkdir -p {args.plot_path}")
# Plotting options
nbinsx = int(args.nbinsx)
nbinsy = int(args.nbinsy)

input_mc = pd.read_parquet(f"{args.inp}/MC_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")

if args.xvar == "corrFactorS2":
    input_mc["corrFactorS2"] = input_mc['w_post_S2_unnorm']/input_mc['w_post_S1']
elif args.xvar == "corrFactorS1":
    input_mc["corrFactorS1"] = input_mc['w_post_S1']/input_mc['weight_norm']

print(f" --> Plotting: {args.xvar} vs {args.yvar}")

if args.xlim == "percentile":
    xlo = np.percentile(input_mc[args.xvar], percentiles[0])
    xhi = np.percentile(input_mc[args.xvar], percentiles[1])
    print(xlo)
    print(xhi)
else:
    xlo = float(re.sub("m","-",args.xlim.split(",")[0]))
    xhi = float(re.sub("m","-",args.xlim.split(",")[1]))

if args.xlim == "percentile":
    ylo = np.percentile(input_mc[args.yvar], percentiles[0])
    yhi = np.percentile(input_mc[args.yvar], percentiles[1])
else:
    ylo = float(re.sub("m","-",args.ylim.split(",")[0]))
    yhi = float(re.sub("m","-",args.ylim.split(",")[1]))


hists = {}

fig, ax = plt.subplots()

hists['mc'] = np.histogram2d(input_mc[args.xvar],input_mc[args.yvar],
    bins = (nbinsx,nbinsy),
    range = ((xlo, xhi), (ylo, yhi))
)

vmin, vmax = 0, hists['mc'][0].max()

mplhep.hist2dplot(hists['mc'],
    cbar = True,
    cbarextend = True,
    cmin = vmin,
    cmax = vmax,
    flow = "none",
    ax = ax
)

ax.set_xlabel(args.xvar)
ax.set_ylabel(args.yvar)

# Add label
mplhep.cms.label(
    f"Preliminary ({detector_name[args.detector]})",
    data = True,
    year = "2022",
    com = "13.6",
    lumi = 26.7,
    lumi_format="{0:.1f}",
    ax = ax,
    fontsize=25
)

if args.ext != "":
    ext_str = f"_{args.ext}"
else:
    ext_str = ""

plt.savefig(f"{args.plot_path}/zee_weight_2d_{args.detector}_{args.dataset}_{args.xvar}_vs_{args.yvar}{ext_str}.pdf")
plt.savefig(f"{args.plot_path}/zee_weight_2d_{args.detector}_{args.dataset}_{args.xvar}_vs_{args.yvar}{ext_str}.png")
