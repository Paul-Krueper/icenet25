import torch 
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# Converting covariance to correlation matrices
# python plot_corr.py -d EEm --dataset test  --plot-path plots_EEm_correlations -i samples_processed_nonorm
def cov_to_corr(cov_matrix):
    stddev = torch.sqrt(torch.diag(cov_matrix))
    stddev_matrix = torch.diag_embed(stddev)
    corr_matrix = torch.inverse(stddev_matrix) @ cov_matrix @ torch.inverse(stddev_matrix)
    return corr_matrix

def cov(x, y,w):
    """Weighted Covariance"""
   
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    
    return cov(x, y, w) / ( np.sqrt(cov(x, x, w) * cov(y, y, w)))

#plot the diference in correlations betwenn mc and data and data and flow
def plot_correlation_matrices(data,mc, shape_var_list, path):

    w_post_S1=mc["w_post_S1"]
    w_post_S2=mc["w_post_S2_unnorm"]
    
    data=data[shape_var_list] #only keep our wanted columns
    mc=mc[shape_var_list]
    
    # define empty correlation matrices of dimension " #columns X #columns "
    correl_data=np.zeros([len(mc.columns),len(mc.columns)])
    correl_mc=np.zeros([len(mc.columns),len(mc.columns)])
    correl_mc_corrected=np.zeros([len(mc.columns),len(mc.columns)])

  
    for  ix,column_x in enumerate(shape_var_list):
        for iy,column_y in enumerate(shape_var_list):


            #calculating the covariance matrices
            
            correl_data[ix][iy]         =  corr(data[column_x],data[column_y],   w=np.ones(len(data[column_x]))) # data has weights = 1
            correl_mc[ix][iy]           = corr( mc[column_x],mc[column_y],       w = abs(w_post_S1) )
            correl_mc_corrected[ix][iy]  = corr( mc[column_x],mc[column_y],      w = abs(w_post_S2) )
            
#     data_cov         = torch.cov( torch.tensor(data.to_numpy()).T  )
#     mc_cov          = torch.cov( torch.tensor(mc.to_numpy()).T           , aweights = torch.Tensor( abs(w_post_S1.to_numpy())  ))
#     mc_corrected_cov = torch.cov( torch.tensor(mc.to_numpy()).T , aweights = torch.Tensor( abs(w_post_S2.to_numpy())  ))
        
#     correl_data = cov_to_corr(data_cov)
#     correl_mc = cov_to_corr(mc_cov)
#     correl_mc_corrected_corr = cov_to_corr(mc_corrected_cov)


    print(correl_data)
        # matrices setup ended! Now plotting part!
    fig, ax = plt.subplots(figsize=(41,41))
    cax = ax.matshow( 100*( correl_data - correl_mc_corrected ), cmap = 'bwr', vmin = -35, vmax = 35)
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 70)
    cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110, labelpad=60)

        # ploting the cov matrix values
    factors_sum = 0
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( correl_data - correl_mc_corrected )):
        mean = mean + abs(z)
        count = count + 1
        factors_sum = factors_sum + abs(z)
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 65)    
        
    ax.yaxis.labelpad = 20
    ax.xaxis.labelpad = 20
    mean = mean/count
        #ax.set_xlabel(r'$100 \cdot (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation^{Corr}}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
    plt.title( r'$\rho$(data) - $\rho$(corrected simulation)' + f' [{args.detector}]', fontweight='bold', fontsize = 130 , pad = 60 )
        
    ax.set_xticks(np.arange(len(shape_var_list)))
    ax.set_yticks(np.arange(len(shape_var_list)))
        
        # Apply the replace method to each element of the list
    cleaned_var_names = [name.replace("probe_", "").replace("raw_", "").replace("trkSum","").replace("ChargedIso","").replace("es","").replace("Cone","").replace("PF","").replace("Over","") for name in shape_var_list]
         
    ax.set_xticklabels(cleaned_var_names, fontsize = 50 , rotation=90 )
    ax.set_yticklabels(cleaned_var_names, fontsize = 50 , rotation=0  )

        # Add text below the plot
    plt.figtext(0.5, 0.04, f'Mean Absolute Sum of Coefficients - {round(factors_sum/(2.*count),2)}', ha="center", fontsize= 85)

    ax.tick_params(axis='both', which='major', pad=30)
    plt.tight_layout()

    plt.savefig(path + f'/correlation_matrix_corrected_{args.detector}.pdf')

        ####################################
        # Nominal MC vs Data
        #####################################
    fig, ax = plt.subplots(figsize=(41,41))
    cax = ax.matshow( 100*( correl_data - correl_mc ), cmap = 'bwr', vmin = -35, vmax = 35)
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 90)
    cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110,labelpad=60)
        
        #ploting the cov matrix values
    factors_sum = 0
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( correl_data - correl_mc )):
        mean = mean + abs(z)
        count = count + 1
        factors_sum = factors_sum + abs(z)
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 55)    
        
    mean = mean/count
        #ax.set_xlabel(r'$100 \cdot  (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
    plt.title( r'$\rho$(data) - $\rho$(nominal simulation)' + f' [{args.detector}]',fontweight='bold', fontsize = 140 , pad = 60 )
        
    ax.set_xticks(np.arange(len(shape_var_list)))
    ax.set_yticks(np.arange(len(shape_var_list)))
            
    ax.set_xticklabels(cleaned_var_names, fontsize = 50 , rotation=90 )
    ax.set_yticklabels(cleaned_var_names, fontsize = 50 , rotation=0  )

        # Add text below the plot
    plt.figtext(0.5, 0.04, f'Mean Absolute Sum of Coefficients - {round(factors_sum/(2.*count),2)}', ha="center", fontsize= 85)

    ax.tick_params(axis='both', which='major', pad=30)
    plt.tight_layout()

    plt.savefig(path + f'/correlation_matrix_nominal_{args.detector}.pdf')



kin_var_list = ['probe_pt', 'probe_eta', 'fixedGridRhoAll']
shape_var_list = ['probe_sieie', 'probe_ecalPFClusterIso', 'probe_trkSumPtHollowConeDR03', 'probe_hcalPFClusterIso', 'probe_pfChargedIso', 'probe_phiWidth', 'probe_trkSumPtSolidConeDR04', 'probe_r9', 'probe_pfChargedIsoWorstVtx', 'probe_s4', 'probe_etaWidth', 'probe_mvaID', 'probe_sieip']
detector_name={"EB":"EB","EEp":"EE+","EEm":"EE-"}
percentiles = (0.5,99.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--plot-path', dest='plot_path', default="plots", help='Path to save plots')
parser.add_argument("-i","--inp",default="samples_processed")
args = parser.parse_args()

if not os.path.isdir(args.plot_path):
        os.system(f"mkdir -p {args.plot_path}")


input_mc = pd.read_parquet(f"{args.inp}/MC_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")
input_data = pd.read_parquet(f"{args.inp}/Data_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")

if args.detector in ['EEp','EEm']:
    shape_var_list.extend(['probe_esEffSigmaRR','probe_esEnergyOverRawE'])
  
    
plot_correlation_matrices(input_data,input_mc,shape_var_list,path=f"{args.plot_path}")