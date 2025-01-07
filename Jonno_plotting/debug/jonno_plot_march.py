import os
import re
from optparse import OptionParser
import json
#python jonno_plot_march.py --data "/vols/cms/pfk18/Csplit_Jsamp_Data_val.parquet" --mc "/vols/cms/pfk18/Csplit_Jsamp_DY_val.parquet" --xvars probe_pfChargedIso   --skip-norm-fix --do-reweight-yields
import pandas as pd
import numpy as np
import awkward as ak

import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
import mplhep 
mplhep.style.use("CMS")

from validation_tools import *
#from hep_ml.reweight import BinsReweighter

fontsize_map = {
    "legend" : 18,
    "x_label": 24,
    "y_label": 24,
    "x_tick_label": 16,
    "y_tick_label": 22,
    "values": 10,
    "values_title": 14,
    "cms_title": 22
}

plot_options_map = {
    "total_linewidth" : 2,
    "data_linewidth" : 2
}

quantile_features = ['probe_pt', 'probe_eta', 'fixedGridRhoAll']
n_quantiles = 20
quantiles = np.linspace(0,1,n_quantiles+1)

def get_options():
    parser = OptionParser()
    parser.add_option('-d','--data', dest='data', default=None, help="Comma-sep list of data parquet files")
    parser.add_option('-s','--mc', dest='mc', default=None, help="Comma-sep list of MC parquet files")
    parser.add_option('-x','--xvars', dest='xvars', default=None, help="Comma-sep list of x variables")
    parser.add_option('-e','--eta-region', dest='eta_region', default="EB", help="Eta-region: [inclusive,EB,EEplus,EEminus]")
    parser.add_option('--plot-path', dest='plot_path', default=".", help="Plot path")
    parser.add_option('--ext', dest='ext', default="", help="Extension for saving")
    parser.add_option('--do-reweight-yields', dest='do_reweight_yields', default=False, action="store_true", help="Reweight MC total yield to data")
    parser.add_option('--do-quantile-rwgt', dest='do_quantile_rwgt', default=False, action="store_true", help="Reweight MC to data in quantile_features using custom code")
    parser.add_option('--do-hepml-rwgt', dest='do_hepml_rwgt', default=False, action="store_true", help="Reweight MC to data in quantile_features using hep_ml package")
    parser.add_option('--do-hepml-rwgt-twostep', dest='do_hepml_rwgt_twostep', default=False, action="store_true", help="Reweight MC to data in quantile_features using hep_ml package with two-step procedure")
    parser.add_option('--do-correction-factor-plot', dest='do_correction_factor_plot', default=False, action="store_true", help="Plot histogram of correction factors")
    parser.add_option('--skip-norm-fix', dest='skip_norm_fix', default=False, action="store_true", help="Skip norm fix")
    parser.add_option('--ratio-lim', dest='ratio_lim', default=None, help="y-axis range e.g. -0.1,2. If None will estimate from data")
    parser.add_option('--figsize', dest='figsize', default="10,12", help="Figure size")
    parser.add_option('--fontsize', dest='fontsize', default="cms_title:22", help="Comma-separated list")
    parser.add_option('--plot-options', dest='plot_options', default="values_title_pad:0.75", help="Comma-separated list")
    parser.add_option('--cms-label', dest='cms_label', default="Work in Progress", help="CMS sub-label")
    return parser.parse_args()
(opt,args) = get_options()

# Change fontsize map according to input
for fontsize_opt in opt.fontsize.split(","):
    fontsize_map[fontsize_opt.split(":")[0]] = float(fontsize_opt.split(":")[1])
# Change plot_options map according to input
for plot_options_opt in opt.plot_options.split(","):
    if plot_options_opt.split(":")[0] == "unit":
        plot_options_map[plot_options_opt.split(":")[0]] = plot_options_opt.split(":")[1]
    else:
        plot_options_map[plot_options_opt.split(":")[0]] = float(plot_options_opt.split(":")[1])

# Load dataframes
dfs = {
    "data" : [],
    "mc" : [],
}

columns_data = ['tag_mvaID', "probe_ScEta",'probe_pt', 'probe_eta', 'fixedGridRhoAll', 'probe_r9', 'probe_sieie', 'probe_etaWidth', 'probe_phiWidth', 'probe_sieip', 'probe_s4', 'probe_ecalPFClusterIso', 'probe_trkSumPtHollowConeDR03', 'probe_trkSumPtSolidConeDR04', 'probe_pfChargedIso', 'probe_pfChargedIsoWorstVtx',  'probe_hcalPFClusterIso', 'probe_mvaID']#,'mass']
columns_mc = ['tag_mvaID',"probe_ScEta", 'probe_pt', 'probe_eta',  'fixedGridRhoAll', 'probe_r9', 'probe_sieie', 'probe_etaWidth', 'probe_phiWidth', 'probe_sieip', 'probe_s4', 'probe_ecalPFClusterIso', 'probe_trkSumPtHollowConeDR03', 'probe_trkSumPtSolidConeDR04', 'probe_pfChargedIso', 'probe_pfChargedIsoWorstVtx', 'probe_hcalPFClusterIso', 'probe_mvaID', 'weight']#,'mass']

for col in columns_mc:
    if col in xvar_input_name_map.keys():
        columns_mc.append(xvar_input_name_map[col])

print(" --> Loading data")
for file_data in opt.data.split(","):
    df = pd.read_parquet(file_data, engine="pyarrow", columns=columns_data)
    # Apply cuts:
    mask = df['tag_mvaID'] > 0.
    if opt.eta_region == "EB":
        mask *= abs(df['probe_ScEta']) < 1.5
    elif opt.eta_region == "EEplus":
        mask *= df['probe_ScEta'] > 1.5
    elif opt.eta_region == "EEminus":
        mask *= df['probe_ScEta'] < -1.5
    # Append to list
    dfs['data'].append(df[mask])
dfs['data'] = pd.concat(dfs['data'], ignore_index=True)
# Add weight column
dfs['data']['weight'] = 1.

print(" --> Loading MC")
for file_mc in opt.mc.split(","):
    df = pd.read_parquet(file_mc, engine="pyarrow", columns=columns_mc)
    # Apply cuts:
    mask = df['tag_mvaID'] > 0.
    mask *= df['fixedGridRhoAll']==df['fixedGridRhoAll']
    if opt.eta_region == "EB":
        mask *= abs(df['probe_ScEta']) < 1.5
    elif opt.eta_region == "EEplus":
        mask *= df['probe_ScEta'] > 1.5
    elif opt.eta_region == "EEminus":
        mask *= df['probe_ScEta'] < -1.5
    # Append to list
    dfs['mc'].append(df[mask])
dfs['mc'] = pd.concat(dfs['mc'], ignore_index=True)

if opt.do_reweight_yields:
    print(" --> Reweighting MC to have same total yield as data")
    dfs['mc']['weight'] = dfs['mc']['weight']*(np.sum(dfs['data']['weight'])/np.sum(dfs['mc']['weight']))

# Custom "conditional" quantile reweighting
if opt.do_quantile_rwgt:
    print(" --> Doing kinematic reweighting (quantiles) for:", quantile_features)
    res = np.array(get_conditional_quantiles(dfs['data'], quantile_features, quantiles))
    yields_data, ind_data = get_yields(dfs['data'], quantile_features, res)
    yields_mc, ind_mc = get_yields(dfs['mc'], quantile_features, res)
    sf = np.array(yields_data)/np.array(yields_mc)
    rwgt = get_reweight_factors(dfs['mc'], sf, ind_mc)
    new_weights = dfs['mc']['weight'] * rwgt
    norm_sf = dfs['mc']['weight'].sum() / new_weights.sum()
    dfs['mc']['quantile_rwgt'] = rwgt*norm_sf

# HEP ML reweighting
if opt.do_hepml_rwgt:
    print(" --> Doing kinematic reweighting (hep_ml) for:", quantile_features)
    reweighter = BinsReweighter(n_bins=n_quantiles, n_neighs=0)
    reweighter.fit(original=dfs['mc'][quantile_features], target=dfs['data'][quantile_features], 
        original_weight=dfs['mc']['weight'], target_weight=dfs['data']['weight']
    )
    rwgt = reweighter.predict_weights(dfs['mc'][quantile_features])
    new_weights = dfs['mc']['weight'] * rwgt
    norm_sf = dfs['mc']['weight'].sum() / new_weights.sum()
    dfs['mc']['hepml_rwgt'] = rwgt*norm_sf

if opt.do_hepml_rwgt_twostep:
    print(" --> Doing kinematic reweighting (hep_ml, 2-step)")
    xa = "fixedGridRhoAll"
    reweighter_a = BinsReweighter(n_bins=3000, n_neighs=0)
    reweighter_a.fit(original=dfs['mc'][xa], target=dfs['data'][xa], 
        original_weight=dfs['mc']['weight'], target_weight=dfs['data']['weight']
    )
    rwgt_a = reweighter_a.predict_weights(dfs['mc'][xa])
    new_weights = dfs['mc']['weight'] * rwgt_a
    norm_sf = dfs['mc']['weight'].sum() / new_weights.sum()
    rwgt_a *= norm_sf
    xb = ['probe_pt','probe_eta']
    if opt.eta_region in ['EEplus','EEminus']:
        reweighter_b = BinsReweighter(n_bins=50, n_neighs=0)
    else:
        reweighter_b = BinsReweighter(n_bins=100, n_neighs=0)
    reweighter_b.fit(original=dfs['mc'][xb], target=dfs['data'][xb],                                
        original_weight=dfs['mc']['weight']*rwgt_a, target_weight=dfs['data']['weight']
    )
    rwgt_b = reweighter_b.predict_weights(dfs['mc'][xb])
    new_weights = dfs['mc']['weight'] * rwgt_a * rwgt_b
    norm_sf = dfs['mc']['weight'].sum() / new_weights.sum()
    dfs['mc']['hepml_rwgt'] = rwgt_a*rwgt_b*norm_sf

                  
# Ensure MC yield remains unchanged after EGM reweighting correction factors
if not opt.skip_norm_fix:
    new_weights = (dfs['mc']['weight'] * dfs['mc']['hepml_rwgt'] * dfs['mc']['probe_egmReweightingFactor'])
    norm_sf = (dfs['mc']['weight'] * dfs['mc']['hepml_rwgt']).sum() / new_weights.sum()
    dfs['mc']['probe_egmReweightingFactor_unnorm'] = dfs['mc']['probe_egmReweightingFactor']
    dfs['mc']['probe_egmReweightingFactor'] = dfs['mc']['probe_egmReweightingFactor']*norm_sf

# Plot correction histogram
if opt.do_correction_factor_plot:
    xvar_name = 'probe_egmReweightingFactor'
    w = dfs['mc']['hepml_rwgt']*dfs['mc']['weight']
    wmean = (dfs['mc'][xvar_name]*w).sum() / w.sum()
    fig, ax = plt.subplots()
    ax.hist(dfs['mc'][xvar_name], bins=50, range=(0,3), weights=w, label=f"$\\eta$-region: {opt.eta_region} (Weighted mean = {wmean:.3f}", color=color_map['mc_nominal'])
    ax.set_ylabel("Weighted events")
    ax.set_xlabel("EGM NN reweight")
    ax.axvline(wmean, color='red', alpha=0.5)
    ax.legend(loc='best', fontsize=fontsize_map['legend'], facecolor='white', edgecolor='black', frameon=True)
    # Add text values to plot 
    mplhep.cms.label(llabel=f"Simulation {opt.cms_label}", ax=ax, rlabel="2022 postEE (13 TeV)", fontsize=0.8*fontsize_map['cms_title'])
    # Save plot 
    ext_str = f".{opt.ext}" if opt.ext != "" else ""
    fig.savefig(f"{opt.plot_path}/tnp.{opt.eta_region}.{xvar_name}{ext_str}.pdf", bbox_inches="tight")
    fig.savefig(f"{opt.plot_path}/tnp.{opt.eta_region}.{xvar_name}{ext_str}.png", bbox_inches="tight")

    ax.cla()

    ax.hist(dfs['mc'][xvar_name], bins=50, range=(0,12), weights=w, label=f"$\\eta$-region: {opt.eta_region} (Weighted mean = {wmean:.3f}", color=color_map['mc_nominal'])
    ax.set_ylabel("Weighted events")
    ax.set_yscale("log")
    ax.set_xlabel("EGM NN reweight")
    ax.axvline(wmean, color='red', alpha=0.5)
    ax.legend(loc='best', fontsize=fontsize_map['legend'], facecolor='white', edgecolor='black', frameon=True)
    # Add text values to plot 
    mplhep.cms.label(llabel=f"Simulation {opt.cms_label}", ax=ax, rlabel="2022 postEE (13 TeV)", fontsize=0.8*fontsize_map['cms_title'])
    # Save plot 
    ext_str = f".{opt.ext}" if opt.ext != "" else ""
    fig.savefig(f"{opt.plot_path}/tnp.{opt.eta_region}.{xvar_name}{ext_str}.log.pdf", bbox_inches="tight")
    fig.savefig(f"{opt.plot_path}/tnp.{opt.eta_region}.{xvar_name}{ext_str}.log.png", bbox_inches="tight")


# Build figure and axes`
axs = []
fig_width, fig_height = float(opt.figsize.split(",")[0]),float(opt.figsize.split(",")[1])
fig = plt.figure(figsize=(fig_width,fig_height))
left, width = 0.1, 0.88
bottom = 0.1
# Ratio plot
height = 0.2
ax = fig.add_axes([left,bottom,width,height])
axs.append(ax)
# Main plot
bottom += height + 0.05
height = 0.63
ax = fig.add_axes([left,bottom,width,height])
axs.append(ax)

# Store chi2_ndof to plot summary
chi2_ndof_summary = {}

for xvar in opt.xvars.split(","):
    print(f" --> Plotting variable: {xvar}")
    bins = xvar_bin_map[xvar]
    if xvar not in ['probe_pt','probe_eta','fixedGridRhoAll']:
        if opt.eta_region in ['EEplus','EEminus']:
            bins = 20
    if xvar in xvar_range_map[opt.eta_region]:
        xrange = xvar_range_map[opt.eta_region][xvar]
    else:
        xrange = (dfs['data'][xvar].min(),dfs['data'][xvar].max())
    do_mc_over_data_ratio = True
    
    # Data histogram
    hists = {}
    hists['data'] = np.histogram(dfs['data'][xvar], bins=bins, range=xrange, weights=dfs['data']['weight'])
    bin_centers = (hists['data'][1][:-1]+hists['data'][1][1:])/2
    bin_widths = (hists['data'][1][1:]-hists['data'][1][:-1])/2
    counts = hists['data'][0]
    
    if xvar in ['probe_pt','probe_eta','fixedGridRhoAll']:
        mc_types = ['pre','nominal','nn']
    else:
        mc_types = ['nominal',"flow"]#,'nn','flow']
    
    weights = {}
    chi2_ndof = {}
    for mc_type in mc_types:
        if mc_type == "pre": 
            xvar_name = translate(xvar, xvar_input_name_map)
            weights[mc_type] = dfs['mc']['weight']
        elif mc_type == "nominal":
            xvar_name = translate(xvar, xvar_input_name_map)
            weights[mc_type] = dfs['mc']['weight']#*dfs['mc']['hepml_rwgt']
        elif mc_type == "nn":
            xvar_name = translate(xvar, xvar_input_name_map)
            weights[mc_type] = dfs['mc']['weight']#*dfs['mc']['hepml_rwgt']*dfs['mc']['probe_egmReweightingFactor']
        elif mc_type == "flow":
            xvar_name = xvar
            weights[mc_type] = dfs['mc']['weight']#*dfs['mc']['hepml_rwgt']
    
        hists[f"mc_{mc_type}"] = axs[1].hist(dfs['mc'][xvar_name], bins=bins, range=xrange, weights=weights[mc_type], color=color_map[f'mc_{mc_type}'], histtype='step')
        hists[f"mc_{mc_type}_sumw2"] = np.histogram(dfs['mc'][xvar_name], bins=bins, range=xrange, weights=weights[mc_type]**2)
 
        # Calculate chi2 and add to dict
        c = (counts-hists[f"mc_{mc_type}"][0])**2 / (hists[f"mc_{mc_type}_sumw2"][0]+counts)
        # Remove Nan: divide by zero error
        c = c[c==c]
        chi2_ndof[f"mc_{mc_type}"] = c.sum() / len(c) 

        # Calculate chi2 w.r.t. nominal stat uncertainty for mc_type == nn
        if mc_type == "nn":
            c = (counts-hists[f"mc_{mc_type}"][0])**2 / (hists[f"mc_nominal_sumw2"][0]+counts)
            c = c[c==c]
            chi2_ndof[f"mc_{mc_type}_rescal"] = c.sum() / len(c)
    
        # Add stat unc boxes
        for i in range(len(bin_widths)):
            point = (bin_centers[i]-bin_widths[i],hists[f"mc_{mc_type}"][0][i]-hists[f"mc_{mc_type}_sumw2"][0][i]**0.5)
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists[f"mc_{mc_type}_sumw2"][0][i]**0.5, color=color_map[f'mc_{mc_type}'], alpha=0.3)
            axs[1].add_patch(rect)
    
        # Add ratio 
        if do_mc_over_data_ratio:
            ratio = hists[f"mc_{mc_type}"][0]/counts
            err = hists[f"mc_{mc_type}_sumw2"][0]**0.5/counts
            for i in range(len(bin_widths)):
                x = (bin_centers[i]-bin_widths[i], bin_centers[i]+bin_widths[i])
                y = (ratio[i],ratio[i])
                axs[0].plot(x, y, color=color_map[f'mc_{mc_type}'])
                point = (bin_centers[i]-bin_widths[i], ratio[i]-err[i])
                rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*err[i], color=color_map[f'mc_{mc_type}'], alpha=0.3)
                axs[0].add_patch(rect)

    axs[1].errorbar(bin_centers, counts, yerr=counts**0.5, ls='None', color=color_map['data'], capsize=2, linewidth=plot_options_map['data_linewidth'], marker='o', markersize=8, label="Data")
    if do_mc_over_data_ratio:
        axs[0].errorbar(bin_centers, np.ones_like(counts), yerr=counts**0.5/counts, ls='None', color=color_map['data'], capsize=2, linewidth=plot_options_map['data_linewidth'], marker='o', markersize=8)
    
    axs[0].set_xlim(xrange)
    axs[1].set_xlim(xrange)
    if xvar in logy_variables:
        axs[1].set_ylim(1e3,axs[1].get_ylim()[1]*2)
        axs[1].set_yscale("log")
    else:
        axs[1].set_ylim(0,axs[1].get_ylim()[1]*1.1)
    axs[0].set_xlabel(translate(xvar,xvar_label_map))
    if do_mc_over_data_ratio:
        axs[0].set_ylabel("MC / data", fontsize=fontsize_map['y_label'], loc='center')
    axs[1].set_ylabel("Events", fontsize=fontsize_map['y_label'])
    if opt.ratio_lim is not None:
        rmin, rmax = float(opt.ratio_lim.split(",")[0]), float(opt.ratio_lim.split(",")[1])
        axs[0].set_ylim(rmin,rmax)
    else:
        axs[0].set_ylim(0.75,1.25)
    axs[0].axhline(1, color='grey', alpha=0.5)
    
    # Add legend
    handles, labels = axs[1].get_legend_handles_labels()
    for mc_type in mc_types:
        line = matplotlib.lines.Line2D([],[], color=color_map[f'mc_{mc_type}'], linewidth=2)
        rect = matplotlib.patches.Rectangle((0,0), 0, 0, facecolor=color_map[f'mc_{mc_type}'], alpha=0.3)
        handle = (line,rect)
        handles.append(handle)
        c = chi2_ndof[f'mc_{mc_type}']
        chi2_str = "$\\chi^2_{d.o.f}$"
        if mc_type == "nn":
            c_rescal = chi2_ndof[f'mc_{mc_type}_rescal']
            labels.append(label_map[mc_type] + f" : {chi2_str} = {c:.2f} ({c_rescal:.2f})")
        else:
            labels.append(label_map[mc_type] + f" : {chi2_str} = {c:.2f}")
    # Add dummy for eta region
    line = matplotlib.lines.Line2D([],[], color="white", linewidth=0)
    handles.append(line)
    eta_region_str = "Z$\\rightarrow$ee (TnP), $\\eta$ region: "
    if opt.eta_region == "inclusive":
        eta_region_str += "inclusive"
    elif opt.eta_region == "EB":
        eta_region_str += "EB"
    elif opt.eta_region == "EEplus":
        eta_region_str += "EE+"
    elif opt.eta_region == "EEminus":
        eta_region_str += "EE-"
    labels.append(eta_region_str)
    axs[1].legend(handles, labels, loc='best', fontsize=fontsize_map['legend'], facecolor='white', edgecolor='black', frameon=True)
    
    # Add text values to plot 
    mplhep.cms.label(llabel=opt.cms_label, ax=axs[1], rlabel="26.67 fb$^{-1}$, 2022 postEE (13 TeV)", fontsize=fontsize_map['cms_title'])
    
    # Save plot 
    ext_str = f".{opt.ext}" if opt.ext != "" else ""
    fig.savefig(f"{opt.plot_path}/tnp.{opt.eta_region}.{xvar}{ext_str}.pdf", bbox_inches="tight")
    fig.savefig(f"{opt.plot_path}/tnp.{opt.eta_region}.{xvar}{ext_str}.png", bbox_inches="tight")

    axs[0].cla()
    axs[1].cla()

    chi2_ndof_summary[xvar] = chi2_ndof

ext_str = f".{opt.ext}" if opt.ext != "" else ""
with open(f"tnp.{opt.eta_region}.chi2_ndof_summary{ext_str}.json", "w") as jf:
    json.dump(chi2_ndof_summary, jf, indent=4)
