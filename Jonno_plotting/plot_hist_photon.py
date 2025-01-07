import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import os

from utils import *

#python plot_hist_photon.py --detector EB --dataset test --do-variance-panel --plot-path plots_photon_EB -i samples_processed_photons


mplhep.style.use("CMS") 
mplhep.style.use({"savefig.bbox": "tight"})

kin_var_list = ['probe_pt', 'probe_eta', 'Rho_fixedGridRhoAll']
shape_var_list = ['probe_sieie', 'probe_ecalPFClusterIso', 'probe_trkSumPtHollowConeDR03', 'probe_hcalPFClusterIso', 'probe_pfChargedIso', 'probe_phiWidth', 'probe_trkSumPtSolidConeDR04', 'probe_r9', 'probe_pfChargedIsoWorstVtx', 'probe_s4', 'probe_etaWidth', 'probe_mvaID', 'probe_sieip']
legend_loc=["upper left","upper right","upper right", "upper right", "upper right", "upper right", "upper right", "upper left", "upper right", "upper left", "upper right", "upper left", "upper left","upper right","upper right","upper right"]
var={"probe_pt":0.55,"probe_eta":0.03,"Rho_fixedGridRhoAll":0.1,'probe_sieie':1, 'probe_ecalPFClusterIso':0.25, 'probe_trkSumPtHollowConeDR03':0.2, 'probe_hcalPFClusterIso':0.1, 'probe_pfChargedIso':0.1, 'probe_phiWidth':0.1, 'probe_trkSumPtSolidConeDR04':2, 'probe_r9':0.1, 'probe_pfChargedIsoWorstVtx':0.3, 'probe_s4':0.1, 'probe_etaWidth':0.1, 'probe_mvaID':0.1, 'probe_sieip':0.1,"probe_esEffSigmaRR":1,"probe_esEnergyOverRawE":0.08}
detector_name={"EB":"EB","EEp":"EE+","EEm":"EE-"}


percentiles = (0.5,99.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--nbins', dest='nbins', default=40, type=int, help='Number of bins in plot')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--do-variance-panel', dest='do_variance_panel', default=False, action="store_true", help='Add variance panel')
parser.add_argument('--plot-path', dest='plot_path', default="plots", help='Path to save plots')
parser.add_argument("-i","--inp",default="samples_processed")
args = parser.parse_args()

# Plotting options
nbins = int(args.nbins)

input_mc = pd.read_parquet(f"{args.inp}/MC_Zmmy_{args.detector}_processed.parquet", engine="pyarrow")
input_data = pd.read_parquet(f"{args.inp}/Data_Zmmy_{args.detector}_processed.parquet", engine="pyarrow")

if args.detector in ['EEp','EEm']:
    shape_var_list.extend(['probe_esEffSigmaRR','probe_esEnergyOverRawE'])
    legend_loc.extend(["upper right","upper right"])

if args.do_variance_panel:
    _, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [1.5, 1, 1], 'hspace': 0}, sharex=True, figsize=(10,14))
else:
    _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1.5, 1], 'hspace': 0}, sharex=True)

for idx,v in enumerate(shape_var_list+kin_var_list):
    
    print(f" --> Plotting: {v}")
    v_corr = f"{v}_corr"
    
    if v == "probe_mvaID":
        lo, hi = (-0.9,1.0)
    elif v in ['probe_ecalPFClusterIso','probe_trkSumPtHollowConeDR03','probe_hcalPFClusterIso','probe_pfChargedIso','probe_trkSumPtSolidConeDR04','probe_pfChargedIsoWorstVtx']:
        lo = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[0])
        hi = np.percentile(pd.concat([input_mc[v],input_data[v]]), 99)
    else:
        lo = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[0])
        hi = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[1])
    
    hists = {}
    
    hists['data'] = np.histogram(input_data[v], nbins, (lo,hi), weights=input_data['weight'])
    hists['data_sumw2'] = np.histogram(input_data[v], nbins, (lo,hi), weights=input_data['weight']**2)
    
    hists['mc'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S1'])
    hists['mc_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S1']**2)
    print(input_mc["w_post_S2_unnorm"].sum()/input_mc["w_post_S2"].sum())
    hists['mc_rwgt'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S2'])
    hists['mc_rwgt_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S2']**2)
    
    if v not in kin_var_list:
        hists['mc_flow'] = np.histogram(input_mc[v_corr], nbins, (lo,hi), weights=input_mc['w_post_S1'])
        hists['mc_flow_sumw2'] = np.histogram(input_mc[v_corr], nbins, (lo,hi), weights=input_mc['w_post_S1']**2)
        
    ind = (hists["data"][0] > 0) & (hists["mc"][0] > 0)
    

    #chi2 = np.sum((counts_mc[ind] - counts_data[ind])**2 / (err_mc[ind]**2 + err_data[ind]**2))

    chi2 = np.sum( (hists["data"][0][ind]-hists["mc"][0][ind])**2/(hists["mc_sumw2"][0][ind]+hists["data"][0][ind]) ) / (len(hists["data"][0][ind]))
    chi2_rwgt = np.sum( (hists["data"][0][ind]-hists["mc_rwgt"][0][ind])**2/(hists["mc_rwgt_sumw2"][0][ind]+hists["data"][0][ind]) ) / (len(hists["data"][0][ind]))
    if v not in kin_var_list:
        chi2_flow = np.sum( (hists["data"][0]-hists["mc_flow"][0])**2/(hists["mc_flow_sumw2"][0]+hists["data"][0]) ) / (len(hists["data"][0][ind]))


    # Top panel
    mplhep.histplot(
        hists['data'],
        w2 = hists['data_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = "Data",
        histtype = "fill",
        flow = "none",
        alpha = 0.5,
        facecolor = '#9c9ca1'
    )
    
    mplhep.histplot(
        hists['mc'],
        w2 = hists['mc_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = f"MC $\chi^2$/{args.nbins}.={chi2:.1f}",
        histtype = "errorbar",
        flow = "none",
        color = '#5790fc'
    )
    
    if v not in kin_var_list:
        mplhep.histplot(
            hists['mc_rwgt'],
            w2 = hists['mc_rwgt_sumw2'][0],
            w2method = "poisson",
            ax = ax[0],
            label = f"MC, rwgt-corrected $\chi^2$/{args.nbins}={chi2_rwgt:.1f} ",
            histtype = "errorbar",
            flow = "none",
            color = '#e42536'
        )
        
        mplhep.histplot(
            hists['mc_flow'],
            w2 = hists['mc_flow_sumw2'][0],
            w2method = "poisson",
            ax = ax[0],
            label = f"MC, flow-corrected $\chi^2$/{args.nbins}={chi2_flow:.1f}",
            histtype = "errorbar",
            flow = "none",
            color = "#ffa90e"
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
            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_sumw2'][0][i]**0.5/hists['mc'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_sumw2'][0][i]**0.5/hists['mc'][0][i], facecolor='#5790fc', alpha=0.1)
            ax[2].add_patch(rect) 

            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_rwgt_sumw2'][0][i]**0.5/hists['mc_rwgt'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_rwgt_sumw2'][0][i]**0.5/hists['mc_rwgt'][0][i], facecolor='#e42536', alpha=0.1)
            ax[2].add_patch(rect) 

            if v not in kin_var_list:
                point = (bin_centers[i]-bin_widths[i], -1*hists['mc_flow_sumw2'][0][i]**0.5/hists['mc_flow'][0][i])
                rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_flow_sumw2'][0][i]**0.5/hists['mc_flow'][0][i], facecolor='#f89c20', alpha=0.1)
                ax[2].add_patch(rect) 


    
    # Ratio panel
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
            (hists['mc'][0]/hists['mc_rwgt'][0],hists['mc'][1]),
            yerr = hists['mc_sumw2'][0]**0.5/hists['mc_rwgt_sumw2'][0],
            ax = ax[1],
            histtype = "errorbar",
            flow = "none",
            color = '#5790fc'
        )
    else:
        mplhep.histplot(
        (hists['mc'][0]/hists['data'][0],hists['mc'][1]),
        yerr = hists['mc_sumw2'][0]**0.5/hists['data'][0],
        ax = ax[1],
        histtype = "errorbar",
        flow = "none",
        color = '#5790fc'
        )
        mplhep.histplot(
            (hists['mc_rwgt'][0]/hists['data'][0],hists['mc_rwgt'][1]),
            yerr = hists['mc_rwgt_sumw2'][0]**0.5/hists['data'][0],
            ax = ax[1],
            histtype = "errorbar",
            flow = "none",
            color = '#e42536'
        )
        
        if v not in kin_var_list:
                mplhep.histplot(
                (hists['mc_flow'][0]/hists['data'][0],hists['mc_flow'][1]),
                yerr = hists['mc_flow_sumw2'][0]**0.5/hists['data'][0],
                ax = ax[1],
                histtype = "errorbar",
                flow = "none",
                color = "#ffa90e"
        )

    # Variance panel
    if args.do_variance_panel:
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
        if v not in kin_var_list:
            mplhep.histplot(
                (-1*hists['mc_flow_sumw2'][0]**0.5/hists['mc_flow'][0],hists['data'][1]),
                ax = ax[2],
                histtype = "step",
                edges = False,
                flow = "none",
                color = "#ffa90e"
            )
            mplhep.histplot(
                (hists['mc_flow_sumw2'][0]**0.5/hists['mc_flow'][0],hists['data'][1]),
                ax = ax[2],
                histtype = "step",
                edges = False,
                flow = "none",
                color = "#ffa90e"
            ) 
    
    ax[0].set_xlim(lo,hi)
    if (v=="probe_hcalPFClusterIso")&(args.detector =="EB"):
        ax[0].set_ylim(2*10**3,3*10**6)
    if (v=="probe_sieip")&(args.detector =="EB"):
        ax[0].set_ylim(0,4*10**5)
    if (v=="probe_sieie")&(args.detector =="EB"):
        ax[0].set_ylim(0,6*10**5)
    if (v=="probe_pfChargedIsoWorstVtx")&(args.detector =="EB"):
        ax[0].set_ylim(8*10**2,1.3*10**6)
    if (v=="probe_esEffSigmaRR")&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(8*10**2,5*10**4)
    if (v=="probe_etaWidth" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(8*10**2,5.5*10**4)
    if (v=="probe_hcalPFClusterIso" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(4*10**2,3*10**5)
    if (v=="probe_r9" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(0,60000)
    if (v=="probe_sieie" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(0,8*10**4)
    if (v=="probe_sieip" )&(args.detector in ['EEp','EEm']):
        ax[0].set_ylim(0,4*10**4)
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
        ax[1].set_ylabel("MC / MC_rwgt", loc='center')
    else:    
        ax[1].set_ylabel("MC / data", loc='center')

    
    ax[0].legend( fontsize=20,loc=legend_loc[idx])

    if v in ['probe_ecalPFClusterIso','probe_trkSumPtHollowConeDR03','probe_hcalPFClusterIso','probe_pfChargedIso','probe_trkSumPtSolidConeDR04','probe_pfChargedIsoWorstVtx']:
        ax[0].set_yscale("log")

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
