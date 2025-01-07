import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import os
from colour import Color

from utils import *

mplhep.style.use("CMS") 
mplhep.style.use({"savefig.bbox": "tight"})

kin_var_list = ['probe_pt', 'probe_eta', 'fixedGridRhoAll']
legend_loc=["upper right", "upper left", "upper right"]
var={"probe_pt":0.55,"probe_eta":0.03,"fixedGridRhoAll":0.1,'probe_sieie':1, 'probe_ecalPFClusterIso':0.25, 'probe_trkSumPtHollowConeDR03':0.2, 'probe_hcalPFClusterIso':0.1, 'probe_pfChargedIso':0.1, 'probe_phiWidth':0.1, 'probe_trkSumPtSolidConeDR04':2, 'probe_r9':0.1, 'probe_pfChargedIsoWorstVtx':0.3, 'probe_s4':0.1, 'probe_etaWidth':0.1, 'probe_mvaID':0.1, 'probe_sieip':0.1,"probe_esEffSigmaRR":1,"probe_esEnergyOverRawE":0.08}
detector_name={"EB":"EB","EEp":"EE+","EEm":"EE-"}
percentiles = (0.5,99.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--nbins', dest='nbins', default=40, type=int, help='Number of bins in plot')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--do-variance-panel', dest='do_variance_panel', default=False, action="store_true", help='Add variance panel')
parser.add_argument('--plot-path', dest='plot_path', default="plots_kin", help='Path to save plots')
parser.add_argument("-i","--inp",default="samples_processed")
args = parser.parse_args()

# Plotting options
nbins = int(args.nbins)

input_mc = pd.read_parquet(f"{args.inp}/MC_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")
input_data = pd.read_parquet(f"{args.inp}/Data_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")

if args.do_variance_panel:
    _, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [1.5, 1, 1], 'hspace': 0}, sharex=True, figsize=(10,14))
else:
    _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1.5, 1], 'hspace': 0}, sharex=True)

for idx,v in enumerate(kin_var_list):
    print(f" --> Plotting: {v}")
    v_corr = f"{v}_corr"
    
    lo = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[0])
    hi = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[1])
    
    hists = {}
    
    hists['data'] = np.histogram(input_data[v], nbins, (lo,hi), weights=input_data['weight'])
    print(hists['data'])
    hists['data_sumw2'] = np.histogram(input_data[v], nbins, (lo,hi), weights=input_data['weight']**2)

    hists['mc_prekin'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['weight_norm'])
    print(hists['mc_prekin'] )
    hists['mc_prekin_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['weight_norm']**2)
    
    hists['mc'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S1'])
    hists['mc_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S1']**2)
    
    hists['mc_rwgt'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S2'])
    hists['mc_rwgt_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S2']**2)
    
    chi2 = np.sum( (hists["data"][0]-hists["mc"][0])**2/(hists["mc_sumw2"][0]+hists["data"][0]) ) / (len(hists["data"][1])-1)
    chi2_rwgt = np.sum( (hists["data"][0]-hists["mc_rwgt"][0])**2/(hists["mc_rwgt_sumw2"][0]+hists["data"][0]) ) / (len(hists["data"][1])-1)
    chi2_pre = np.sum( (hists["data"][0]-hists["mc_prekin"][0])**2/(hists["mc_prekin_sumw2"][0]+hists["data"][0]) ) / (len(hists["data"][1])-1)

    # Top panel
    mplhep.histplot(
        hists['data'],
        w2 = hists['data_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = "Data",
        histtype = "fill",
        flow = "none",
        alpha = 0.25,
        facecolor = '#9c9ca1'
    )

    mplhep.histplot(
        hists['mc_prekin'],
        w2 = hists['mc_prekin_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = f"MC (pre kinematic rwgt) $\chi^2$/{args.nbins}={chi2_pre:.1f}",
        histtype = "errorbar",
        flow = "none",
        color="#e76300",
        marker="x"
      #  color = "#ffa90e"
    )
    
    mplhep.histplot(
        hists['mc'],
        w2 = hists['mc_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = f"MC $\chi^2$/{args.nbins}={chi2:.1f}",
        histtype = "errorbar",
        flow = "none",
        color = '#5790fc'
    )
    
    mplhep.histplot(
        hists['mc_rwgt'],
        w2 = hists['mc_rwgt_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = f"MC, rwgt-corrected $\chi^2$/{args.nbins}={chi2_rwgt:.1f}",
        histtype = "errorbar",
        flow = "none",
        color = '#e42536'
    )
    
    
    bin_centers = (hists['data'][1][:-1]+hists['data'][1][1:])/2
    bin_widths = (hists['data'][1][1:]-hists['data'][1][:-1])/2

    # Add stat uncertainty boxes
    for i in range(len(bin_widths)):
        point = (bin_centers[i]-bin_widths[i], hists['data'][0][i]-hists['data_sumw2'][0][i]**0.5)
        rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['data_sumw2'][0][i]**0.5, edgecolor='black', facecolor='None', hatch='XX')
        ax[0].add_patch(rect)
    
        point = (bin_centers[i]-bin_widths[i], 1-hists['data_sumw2'][0][i]**0.5/hists['data'][0][i])
        rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['data_sumw2'][0][i]**0.5/hists['data'][0][i], facecolor='#9c9ca1', alpha=0.25, hatch='XX')
        ax[1].add_patch(rect)

        if args.do_variance_panel:
            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_prekin_sumw2'][0][i]**0.5/hists['mc_prekin'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_prekin_sumw2'][0][i]**0.5/hists['mc_prekin'][0][i], facecolor='#a96b59', alpha=0.1)
            ax[2].add_patch(rect)

            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_sumw2'][0][i]**0.5/hists['mc'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_sumw2'][0][i]**0.5/hists['mc'][0][i], facecolor='#5790fc', alpha=0.1)
            ax[2].add_patch(rect)

            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_rwgt_sumw2'][0][i]**0.5/hists['mc_rwgt'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_rwgt_sumw2'][0][i]**0.5/hists['mc_rwgt'][0][i], facecolor='#e42536', alpha=0.1)
            ax[2].add_patch(rect)

    
    # Bottom panel
    mplhep.histplot(
        (1+hists['data_sumw2'][0]**0.5/hists['data'][0],hists['data'][1]),
        ax = ax[1],
        histtype = "step",
        edges = False,
        flow = "none",
        color = 'black'
    )
    
    mplhep.histplot(
        (1-hists['data_sumw2'][0]**0.5/hists['data'][0],hists['data'][1]),
        ax = ax[1],
        histtype = "step",
        edges = False,
        flow = "none",
        color = 'black'
    )
    
    if (v=="probe_eta")|(v=="probe_pt")|(v=="fixedGridRhoAll"):
        mplhep.histplot(
            (hists['mc_rwgt'][0]/hists['mc'][0],hists['mc_rwgt'][1]),
            yerr = hists["mc_rwgt"][0]*hists['mc_rwgt_sumw2'][0]**0.5/hists['mc'][0]**2, #1+abs((MC_old/MC)*(np.sqrt(MC2)/MC)
            ax = ax[1],
            histtype = "errorbar",
            flow = "none",
            color = '#e42536'
        )
   
    
    
    # Variance panel
    if args.do_variance_panel:
        mplhep.histplot(
            (-1*hists['mc_prekin_sumw2'][0]**0.5/hists['mc_prekin'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color  = "#e76300"
        )
        mplhep.histplot(
            (hists['mc_prekin_sumw2'][0]**0.5/hists['mc_prekin'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color  = "#e76300"
        )

        mplhep.histplot(
            (-1*hists['mc_sumw2'][0]**0.5/hists['mc'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#5790fc'
        )
        mplhep.histplot(
            (hists['mc_sumw2'][0]**0.5/hists['mc'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#5790fc'
        )
        mplhep.histplot(
            (-1*hists['mc_rwgt_sumw2'][0]**0.5/hists['mc_rwgt'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#e42536'
        )
        mplhep.histplot(
            (hists['mc_rwgt_sumw2'][0]**0.5/hists['mc_rwgt'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#e42536'
        )
    
    
    
    ax[0].set_xlim(lo,hi)
    ax[1].set_xlim(lo,hi)
    ax[1].set_ylim(0.78,1.22)
    ax[1].axhline(1, color='grey', ls='--')
    
    if args.do_variance_panel:
        ax[2].set_xlim(lo,hi)
        ax[2].set_ylim(-1*var[v],var[v])
        ax[2].set_xlabel(var_name_pretty[v])
        ax[2].set_ylabel("$\\pm \\frac{\\sqrt{\\sum{w_i^2}}}{\\sum{w_i}}$", loc='center')
        ax[2].axhline(0, color='grey', ls='--')
    else:
        ax[1].set_xlabel(var_name_pretty[v])

    ax[0].set_ylabel("Events")
    if (v=="probe_eta")|(v=="probe_pt")|(v=="fixedGridRhoAll"):
        ax[1].set_ylabel("MC_rwgt / MC", loc='center')
    else:    
        ax[1].set_ylabel("MC / data", loc='center')
    
    ax[0].legend(loc=legend_loc[idx], fontsize=20)

    if v == "probe_pt":
        ax[0].set_yscale("log")
    if (v=="probe_eta" )&(args.detector=="EB"):
        ax[0].set_ylim(0,1.2*10**5)
    if (v=="probe_pt" )&(args.detector=="EB"):
        ax[0].set_ylim(0,1.2*10**6)
    if (v=="probe_pt" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(10,1*10**6)
    if (v=="probe_eta" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(0,2*10**4)
    if (v=="fixedGridRhoAll" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(0,4*10**4)
    if (v=="fixedGridRhoAll" )&(args.detector =="EB"):
        ax[0].set_ylim(0,2.5*10**5)
    elif v == "probe_eta":
        ax[0].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1]*1.2)

    for be in (bin_centers-bin_widths):
        ax[1].axvline(be, color='grey', alpha=0.1)
        if args.do_variance_panel:
            ax[2].axvline(be, color='grey', alpha=0.1)
    
    # Add label
    mplhep.cms.label(
        f"Preliminary ({detector_name[args.detector]})",
        data = True,
        year = "2022",
        com = "13.6",
        lumi = 26.7,
        lumi_format="{0:.1f}",
        ax = ax[0],
        fontsize=25
    )
    
    if not os.path.isdir(args.plot_path):
        os.system(f"mkdir -p {args.plot_path}")

    if args.ext != "":
        ext_str = f"_{args.ext}"
    else:
        ext_str = ""

    plt.savefig(f"{args.plot_path}/zee_{args.detector}_{args.dataset}_{v}{ext_str}.pdf")
    plt.savefig(f"{args.plot_path}/zee_{args.detector}_{args.dataset}_{v}{ext_str}.png")
    
    ax[0].cla()
    ax[1].cla()
    if args.do_variance_panel:
        ax[2].cla()