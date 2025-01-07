import pandas as pd
import numpy as np

def unbound_list(x):
    x[0], x[-1] = -np.inf, np.inf
    return x


def get_conditional_quantiles(df, features, quantiles, history=[]):
    if len(features) == 1:
        res = []
        x_quantiles = unbound_list( list(df[features[0]].quantile(q=quantiles)) )
        for i in range(len(quantiles)-1):
            res.append( history+[x_quantiles[i],x_quantiles[i+1]] )
        return res
    else:
        res = []
        x_quantiles = unbound_list( list(df[features[0]].quantile(q=quantiles)) )
        for i in range(len(quantiles)-1):
            mask = (df[features[0]]>=x_quantiles[i])&(df[features[0]]<x_quantiles[i+1])
            res.append( get_conditional_quantiles(df[mask], features[1:], quantiles, history=history+[x_quantiles[i],x_quantiles[i+1]]) )
        return res


def get_yields(df, features, mask_list):
    if len(features) == 1:
        yields, ind = [], []
        for i in range(len(mask_list)):
            mask_info = mask_list[i]
            lo, hi = mask_info[-2], mask_info[-1]
            mask = (df[features[0]]>=lo)&(df[features[0]]<hi)
            yields.append( df[mask]['weight'].sum() )
            ind.append(df[mask].index)
        return yields, ind
    else:
        yields, ind = [], []
        for i in range(len(mask_list)):
            mask_list_new = mask_list[i]
            lo, hi = mask_list_new.flatten()[-2*len(features)], mask_list_new.flatten()[-2*len(features)+1]
            mask = (df[features[0]]>=lo)&(df[features[0]]<hi)
            result = get_yields(df[mask], features[1:], mask_list_new)
            yields.append( result[0] )
            ind.append( result[1] )
        return yields, ind


def get_reweight_factors(df, scale_factors, index):
    rwgt = np.zeros_like(df['weight'])
    scale_factors = scale_factors.flatten()
    index = np.array(index).flatten()
    for i in range(len(scale_factors)):
        sf = scale_factors[i]
        ind = np.array(index[i])
        np.put(rwgt, ind, sf)
    return rwgt


def translate(name, ndict):
    return ndict[name] if name in ndict else name


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
color_map = {
    "mc_pre" : "forestgreen",
    "mc_nominal" : "grey",
    "mc_nn" : "red",
    "mc_flow" : "royalblue",
    "data" : "black"
}

label_map = {
    "pre" : "MC (pre-kin rwgt)",
    "nominal" : "MC",
    "nn" : "MC (EGM NN)",
    "flow" : "MC (flow)"
}

xvar_label_map = {
    "mass":"Zee mass",
    "probe_pt" : "Probe $p_T$ (GeV)",
    "probe_eta" : "Probe $\\eta$",
    "fixedGridRhoAll" : "$\\rho$",
    "probe_r9" : "Probe R$_{9}$",
    "probe_sieie" : "Probe $\\sigma_{i{\\eta}i{\\eta}}$",
    "probe_etaWidth" : "Probe $\\eta$ width",
    "probe_phiWidth" : "Probe $\\phi$ width",
    "probe_sieip" : "Probe $\\sigma_{i{\\eta}i{\\phi}}$",
    "probe_s4" : "Probe S$_{4}$",
    "probe_hoe" : "Probe H/E",
    "probe_ecalPFClusterIso" : "Probe ECALPFClusterIso",
    "probe_trkSumPtHollowConeDR03" : "Probe trkSumPtHollowConeDR03",
    "probe_trkSumPtSolidConeDR04" : "Probe trkSumPtHollowConeDR04",
    "probe_pfChargedIso" : "Probe PFChargedIso",
    "probe_pfChargedIsoWorstVtx" : "Probe PFChargedIso (worst vtx)",
    "probe_esEffSigmaRR" : "Probe esEffSigmaRR",
    "probe_esEnergyOverRawE" : "Probe esEnergyOverRawE",
    "probe_hcalPFClusterIso" : "Probe HCALPFClusterIso",
    "probe_mvaID" : "Probe mvaID",
    "probe_energyErr" : "Probe $\\sigma_{E}$",
    "photon_pt" : "Photon $p_T$ (GeV)",
    "photon_eta" : "Photon $\\eta$",
    "Rho_fixedGridRhoAll" : "$\\rho$",
    "photon_r9" : "Photon R$_{9}$",
    "photon_sieie" : "Photon $\\sigma_{i{\\eta}i{\\eta}}$",
    "photon_etaWidth" : "Photon $\\eta$ width",
    "photon_phiWidth" : "Photon $\\phi$ width",
    "photon_sieip" : "Photon $\\sigma_{i{\\eta}i{\\phi}}$",
    "photon_s4" : "Photon S$_{4}$",
    "photon_hoe" : "Photon H/E",
    "photon_ecalPFClusterIso" : "Photon ECALPFClusterIso",
    "photon_trkSumPtHollowConeDR03" : "Photon trkSumPtHollowConeDR03",
    "photon_trkSumPtSolidConeDR04" : "Photon trkSumPtHollowConeDR04",
    "photon_pfChargedIso" : "Photon PFChargedIso",
    "photon_pfChargedIsoWorstVtx" : "Photon PFChargedIso (worst vtx)",
    "photon_esEffSigmaRR" : "Photon esEffSigmaRR",
    "photon_esEnergyOverRawE" : "Photon esEnergyOverRawE",
    "photon_hcalPFClusterIso" : "Photon HCALPFClusterIso",
    "photon_mvaID" : "Photon mvaID",
    "photon_energyErr" : "Photon $\\sigma_{E}$",
    "lead_pt" : "Lead $p_T$ (GeV)",
    "lead_eta" : "Lead $\\eta$",
    "lead_r9" : "Lead R$_{9}$",
    "lead_sieie" : "Lead $\\sigma_{i{\\eta}i{\\eta}}$",
    "lead_etaWidth" : "Lead $\\eta$ width",
    "lead_phiWidth" : "Lead $\\phi$ width",
    "lead_sieip" : "Lead $\\sigma_{i{\\eta}i{\\phi}}$",
    "lead_s4" : "Lead S$_{4}$",
    "lead_hoe" : "Lead H/E",
    "lead_ecalPFClusterIso" : "Lead ECALPFClusterIso",
    "lead_trkSumPtHollowConeDR03" : "Lead trkSumPtHollowConeDR03",
    "lead_trkSumPtSolidConeDR04" : "Lead trkSumPtHollowConeDR04",
    "lead_pfChargedIso" : "Lead PFChargedIso",
    "lead_pfChargedIsoWorstVtx" : "Lead PFChargedIso (worst vtx)",
    "lead_esEffSigmaRR" : "Lead esEffSigmaRR",
    "lead_esEnergyOverRawE" : "Lead esEnergyOverRawE",
    "lead_hcalPFClusterIso" : "Lead HCALPFClusterIso",
    "lead_mvaID" : "Lead mvaID",
    "lead_energyErr" : "Lead $\\sigma_{E}$",
    "sublead_pt" : "Sublead $p_T$ (GeV)",
    "sublead_eta" : "Sublead $\\eta$",
    "sublead_r9" : "Sublead R$_{9}$",
    "sublead_sieie" : "Sublead $\\sigma_{i{\\eta}i{\\eta}}$",
    "sublead_etaWidth" : "Sublead $\\eta$ width",
    "sublead_phiWidth" : "Sublead $\\phi$ width",
    "sublead_sieip" : "Sublead $\\sigma_{i{\\eta}i{\\phi}}$",
    "sublead_s4" : "Sublead S$_{4}$",
    "sublead_hoe" : "Sublead H/E",
    "sublead_ecalPFClusterIso" : "Sublead ECALPFClusterIso",
    "sublead_trkSumPtHollowConeDR03" : "Sublead trkSumPtHollowConeDR03",
    "sublead_trkSumPtSolidConeDR04" : "Sublead trkSumPtHollowConeDR04",
    "sublead_pfChargedIso" : "Sublead PFChargedIso",
    "sublead_pfChargedIsoWorstVtx" : "Sublead PFChargedIso (worst vtx)",
    "sublead_esEffSigmaRR" : "Sublead esEffSigmaRR",
    "sublead_esEnergyOverRawE" : "Sublead esEnergyOverRawE",
    "sublead_hcalPFClusterIso" : "Sublead HCALPFClusterIso",
    "sublead_mvaID" : "Sublead mvaID",
    "sublead_energyErr" : "Sublead $\\sigma_{E}$"
}

xvar_bin_map = {
    "mass": 40,
    "probe_pt" : 40,
    "probe_eta" : 40,
    "fixedGridRhoAll" : 40,
    "probe_r9" : 40,
    "probe_sieie" : 40,
    "probe_etaWidth" : 40,
    "probe_phiWidth" : 40,
    "probe_sieip" : 40,
    "probe_s4" : 40,
    "probe_hoe" : 40,
    "probe_ecalPFClusterIso" : 40,
    "probe_trkSumPtHollowConeDR03" : 40,
    "probe_trkSumPtSolidConeDR04" : 40,
    "probe_pfChargedIso" : 40,
    "probe_pfChargedIsoWorstVtx" : 40,
    "probe_esEffSigmaRR" : 40,
    "probe_esEnergyOverRawE" : 40,
    "probe_hcalPFClusterIso" : 40, 
    "probe_mvaID" : 40,
    "probe_energyErr" : 40,
    "photon_pt" : 20,
    "photon_eta" : 20,
    "Rho_fixedGridRhoAll" : 20,
    "photon_r9" : 20,
    "photon_sieie" : 20,
    "photon_etaWidth" : 20,
    "photon_phiWidth" : 20,
    "photon_sieip" : 20,
    "photon_s4" : 20,
    "photon_hoe" : 20,
    "photon_ecalPFClusterIso" : 20,
    "photon_trkSumPtHollowConeDR03" : 20,
    "photon_trkSumPtSolidConeDR04" : 20,
    "photon_pfChargedIso" : 20,
    "photon_pfChargedIsoWorstVtx" : 20,
    "photon_esEffSigmaRR" : 20,
    "photon_esEnergyOverRawE" : 20,
    "photon_hcalPFClusterIso" : 20, 
    "photon_mvaID" : 20,
    "photon_energyErr" : 20
}

xvar_range_map = {
    "EB": {
        "mass":(80,100),
        "probe_pt" : (20,100),
        "probe_eta" : (-1.5,1.5),
        "fixedGridRhoAll" : (0,60),
        "probe_r9" : (0.5,1.05),
        "probe_sieie" : (0.006, 0.012),
        "probe_etaWidth" : (0.003, 0.0125),
        "probe_phiWidth" : (0, 0.1),
        "probe_sieip" : (-0.00005,0.00005),
        "probe_s4" : (0.4,1),
        "probe_hoe" : (0, 0.08),
        "probe_ecalPFClusterIso" : (0,3),
        "probe_trkSumPtHollowConeDR03" : (0,4.8),
        "probe_trkSumPtSolidConeDR04" : (0,6.75),
        "probe_pfChargedIso" : (0,5),
        "probe_pfChargedIsoWorstVtx" : (0,50),
        "probe_esEffSigmaRR" : (0,14),
        "probe_esEnergyOverRawE" : (0,0.3),
        "probe_hcalPFClusterIso" : (0,12),
        "probe_mvaID" : (-1,1),
        "probe_energyErr" : (0,3),
        "photon_pt" : (20,100),
        "photon_eta" : (-1.5,1.5),
        "Rho_fixedGridRhoAll" : (0,60),
        "photon_r9" : (0.5,1.05),
        "photon_sieie" : (0.006, 0.012),
        "photon_etaWidth" : (0.003, 0.0125),
        "photon_phiWidth" : (0, 0.1),
        "photon_sieip" : (-0.00005,0.00005),
        "photon_s4" : (0.4,1),
        "photon_hoe" : (0, 0.08),
        "photon_ecalPFClusterIso" : (0,3),
        "photon_trkSumPtHollowConeDR03" : (0,4.8),
        "photon_trkSumPtSolidConeDR04" : (0,6.75),
        "photon_pfChargedIso" : (0,5),
        "photon_pfChargedIsoWorstVtx" : (0,50),
        "photon_esEffSigmaRR" : (0,14),
        "photon_esEnergyOverRawE" : (0,0.3),
        "photon_hcalPFClusterIso" : (0,12),
        "photon_mvaID" : (-1,1),
        "photon_energyErr" : (0,3)
    },
    "EEplus": {
         "mass":(80,100),
        "probe_pt" : (20,100),
        "probe_eta" : (1.5,2.7),
        "fixedGridRhoAll" : (0,60),
        "probe_r9" : (0.7,1.2),
        "probe_sieie" : (0.01, 0.04),
        "probe_etaWidth" : (0, 0.035),
        "probe_phiWidth" : (0, 0.1),
        "probe_sieip" : (-0.0005,0.0005),
        "probe_s4" : (0.4,1.),
        "probe_hoe" : (0, 0.08),
        "probe_ecalPFClusterIso" : (0,3),
        "probe_trkSumPtHollowConeDR03" : (0,4.8),
        "probe_trkSumPtSolidConeDR04" : (0,6.75),
        "probe_pfChargedIso" : (0,5),
        "probe_pfChargedIsoWorstVtx" : (0,50),
        "probe_esEffSigmaRR" : (0,12),
        "probe_esEnergyOverRawE" : (0,0.3),
        "probe_hcalPFClusterIso" : (0,12),
        "probe_mvaID" : (-1,1),
        "probe_energyErr" : (0,8),
        "photon_pt" : (20,100),
        "photon_eta" : (1.5,2.7),
        "Rho_fixedGridRhoAll" : (0,60),
        "photon_r9" : (0.7,1.2),
        "photon_sieie" : (0.01, 0.04),
        "photon_etaWidth" : (0, 0.035),
        "photon_phiWidth" : (0, 0.1),
        "photon_sieip" : (-0.0005,0.0005),
        "photon_s4" : (0.4,1.),
        "photon_hoe" : (0, 0.08),
        "photon_ecalPFClusterIso" : (0,3),
        "photon_trkSumPtHollowConeDR03" : (0,4.8),
        "photon_trkSumPtSolidConeDR04" : (0,6.75),
        "photon_pfChargedIso" : (0,5),
        "photon_pfChargedIsoWorstVtx" : (0,50),
        "photon_esEffSigmaRR" : (0,12),
        "photon_esEnergyOverRawE" : (0,0.3),
        "photon_hcalPFClusterIso" : (0,12),
        "photon_mvaID" : (-1,1),
        "photon_energyErr" : (0,8)
    },
    "EEminus": {
         "mass":(80,100),
        "probe_pt" : (20,100),
        "probe_eta" : (-2.7,-1.5),
        "fixedGridRhoAll" : (0,60),
        "probe_r9" : (0.7,1.2),
        "probe_sieie" : (0.01, 0.04),
        "probe_etaWidth" : (0, 0.035),
        "probe_phiWidth" : (0, 0.1),
        "probe_sieip" : (-0.0005,0.0005),
        "probe_s4" : (0.4,1.),
        "probe_hoe" : (0, 0.08),
        "probe_ecalPFClusterIso" : (0,3),
        "probe_trkSumPtHollowConeDR03" : (0,4.8),
        "probe_trkSumPtSolidConeDR04" : (0,6.75),
        "probe_pfChargedIso" : (0,5),
        "probe_pfChargedIsoWorstVtx" : (0,50),
        "probe_esEffSigmaRR" : (0,12),
        "probe_esEnergyOverRawE" : (0,0.3),
        "probe_hcalPFClusterIso" : (0,12),
        "probe_mvaID" : (-1,1),
        "probe_energyErr" : (0,8),
        "photon_pt" : (20,100),
        "photon_eta" : (-2.7,-1.5),
        "Rho_fixedGridRhoAll" : (0,60),
        "photon_r9" : (0.7,1.2),
        "photon_sieie" : (0.01, 0.04),
        "photon_etaWidth" : (0, 0.035),
        "photon_phiWidth" : (0, 0.1),
        "photon_sieip" : (-0.0005,0.0005),
        "photon_s4" : (0.4,1.),
        "photon_hoe" : (0, 0.08),
        "photon_ecalPFClusterIso" : (0,3),
        "photon_trkSumPtHollowConeDR03" : (0,4.8),
        "photon_trkSumPtSolidConeDR04" : (0,6.75),
        "photon_pfChargedIso" : (0,5),
        "photon_pfChargedIsoWorstVtx" : (0,50),
        "photon_esEffSigmaRR" : (0,12),
        "photon_esEnergyOverRawE" : (0,0.3),
        "photon_hcalPFClusterIso" : (0,12),
        "photon_mvaID" : (-1,1),
        "photon_energyErr" : (0,8)
    }
}

xvar_input_name_map = {
    "probe_r9" : "probe_raw_r9",
    "probe_sieie" : "probe_raw_sieie",
    "probe_etaWidth" : "probe_raw_etaWidth",
    "probe_phiWidth" : "probe_raw_phiWidth",
    "probe_sieip" : "probe_raw_sieip",
    "probe_s4" : "probe_raw_s4",
    "probe_hoe" : "probe_raw_hoe",
    "probe_ecalPFClusterIso" : "probe_raw_ecalPFClusterIso",
    "probe_trkSumPtHollowConeDR03" : "probe_raw_trkSumPtHollowConeDR03",
    "probe_trkSumPtSolidConeDR04" : "probe_raw_trkSumPtSolidConeDR04",
    "probe_pfChargedIso" : "probe_raw_pfChargedIso",
    "probe_pfChargedIsoWorstVtx" : "probe_raw_pfChargedIsoWorstVtx",
    "probe_esEffSigmaRR" : "probe_raw_esEffSigmaRR",
    "probe_esEnergyOverRawE" : "probe_raw_esEnergyOverRawE",
    "probe_hcalPFClusterIso" : "probe_raw_hcalPFClusterIso",
    "probe_mvaID" : "probe_mvaID_nano",
    "probe_energyErr" : "probe_raw_energyErr",
    "photon_r9" : "photon_raw_r9",
    "photon_sieie" : "photon_raw_sieie",
    "photon_etaWidth" : "photon_raw_etaWidth",
    "photon_phiWidth" : "photon_raw_phiWidth",
    "photon_sieip" : "photon_raw_sieip",
    "photon_s4" : "photon_raw_s4",
    "photon_hoe" : "photon_raw_hoe",
    "photon_ecalPFClusterIso" : "photon_raw_ecalPFClusterIso",
    "photon_trkSumPtHollowConeDR03" : "photon_raw_trkSumPtHollowConeDR03",
    "photon_trkSumPtSolidConeDR04" : "photon_raw_trkSumPtSolidConeDR04",
    "photon_pfChargedIso" : "photon_raw_pfChargedIso",
    "photon_pfChargedIsoWorstVtx" : "photon_raw_pfChargedIsoWorstVtx",
    "photon_esEffSigmaRR" : "photon_raw_esEffSigmaRR",
    "photon_esEnergyOverRawE" : "photon_raw_esEnergyOverRawE",
    "photon_hcalPFClusterIso" : "photon_raw_hcalPFClusterIso",
    "photon_mvaID" : "photon_mvaID_raw",
    "photon_energyErr" : "photon_raw_energyErr",
    "r9" : "raw_r9",
    "sieie" : "raw_sieie",
    "etaWidth" : "raw_etaWidth",
    "phiWidth" : "raw_phiWidth",
    "sieip" : "raw_sieip",
    "s4" : "raw_s4",
    "hoe" : "raw_hoe",
    "ecalPFClusterIso" : "raw_ecalPFClusterIso",
    "trkSumPtHollowConeDR03" : "raw_trkSumPtHollowConeDR03",
    "trkSumPtSolidConeDR04" : "raw_trkSumPtSolidConeDR04",
    "pfChargedIso" : "raw_pfChargedIso",
    "pfChargedIsoWorstVtx" : "raw_pfChargedIsoWorstVtx",
    "esEffSigmaRR" : "raw_esEffSigmaRR",
    "esEnergyOverRawE" : "raw_esEnergyOverRawE",
    "hcalPFClusterIso" : "raw_hcalPFClusterIso",
    "mvaID" : "mvaID_nano",
    "energyErr" : "raw_energyErr",

}

logy_variables = ["probe_ecalPFClusterIso","probe_trkSumPtHollowConeDR03","probe_trkSumPtSolidConeDR04","probe_pfChargedIso","probe_pfChargedIsoWorstVtx","probe_esEffSigmaRR","probe_esEnergyOverRawE","probe_hcalPFClusterIso","photon_ecalPFClusterIso","photon_trkSumPtHollowConeDR03","photon_trkSumPtSolidConeDR04","photon_pfChargedIso","photon_pfChargedIsoWorstVtx","photon_esEffSigmaRR","photon_esEnergyOverRawE","photon_hcalPFClusterIso"]