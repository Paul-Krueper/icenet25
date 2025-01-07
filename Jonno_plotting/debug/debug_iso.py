import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
#path_MC="/vols/cms/pfk18/Csplit_Jsamp_DY_val.parquet"
#path_data="/vols/cms/pfk18/Csplit_Jsamp_Data_val.parquet"
path_MC="/vols/cms/pfk18/icenet_files/processed_29_nov_24/MC_val_EB.parquet"
path_data="/vols/cms/pfk18/icenet_files/processed_29_nov_24/Data_val_EB.parquet"
MC=pd.read_parquet(path_MC)
for i in MC.columns:
    print(i)
print("data")

data=pd.read_parquet(path_data)
for i in data.columns:
    print(i)
    
print(data["probe_pfChargedIso"].head(5))
print(data["probe_pfChargedIso"].head(5))
print(MC["probe_pfChargedIso"].head(5))
print(MC["probe_pfChargedIso"].head(5))

data=data[(data.probe_eta<1.444)&(data.probe_eta>-1.444)]
MC=MC[(MC.probe_eta<1.444)&(MC.probe_eta>-1.444)]

ratio=MC.weight.sum() / len(data)
MC.weight*=1/ratio

print(MC.weight.sum() / len(data))

MC_val,edges=np.histogram(MC["probe_pfChargedIso"],bins=30,range=(0,4),weights=MC.weight)
MC_NF_val,edges=np.histogram(MC["probe_pfChargedIso_corr"],bins=30,range=(0,4),weights=MC.weight)
#MC_val,edges=np.histogram(MC["probe_pfChargedIso"],bins=30,range=(0,4),weights=MC.weight)
#MC_NF_val,edges=np.histogram(MC["probe_pfChargedIso_corr"],bins=30,range=(0,4),weights=MC.weight)

data_val,edges=np.histogram(data["probe_pfChargedIso"],bins=30,range=(0,4))


f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
axs[0].hist(edges[:-1],edges,histtype="step",label="MC raw",weights=MC_val)
#xs[0].hist(edges[:-1],edges,histtype="step",label="MC NF",weights=MC_NF_val)
axs[0].hist(edges[:-1],edges,histtype="step",label="data",weights=data_val)
axs[0].legend()
axs[0].set_xlabel("pfChargedIso (validation set)")
#axs[0].set_yscale("log")
axs[0].set_xlim(0,4)
axs[1].scatter(edges[:-1],MC_val/data_val,label="MC")
#axs[1].scatter(edges[:-1],MC_NF_val/data_val,label="MC NF")
axs[1].set_ylabel="MC/data"
axs[1].set_ylim(0.9,1.1)
axs[1].hlines(y=1,xmin=0,xmax=4)
plt.savefig("isoplot_afterNov24processing.pdf")

