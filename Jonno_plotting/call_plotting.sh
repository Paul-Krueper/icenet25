python plot_hist.py -d EB --dataset test --do-variance-panel --plot-path plots_EB -i samples_processed_nonorm
python plot_hist.py -d EEm --dataset test --do-variance-panel --plot-path plots_EEm -i samples_processed_nonorm
python plot_hist.py -d EEp --dataset test --do-variance-panel --plot-path plots_EEp -i samples_processed_nonorm

python plot_kin_hist.py -d EB --dataset test --do-variance-panel --plot-path plots_kin_EB -i samples_processed_nonorm
python plot_kin_hist.py -d EEm --dataset test --do-variance-panel --plot-path plots_kin_EEm -i samples_processed_nonorm
python plot_kin_hist.py -d EEp --dataset test --do-variance-panel --plot-path plots_kin_EEp -i samples_processed_nonorm


python plot_weight_dist.py -d EB --dataset test --plot-path plots_EB_w -i samples_processed_nonorm
python plot_weight_dist.py -d EB --dataset test --plot-path plots_EB_cor --do-correction-factor -i samples_processed_nonorm
python plot_weight_dist.py -d EEp --dataset test --plot-path plots_EEp_w -i samples_processed_nonorm
python plot_weight_dist.py -d EEp --dataset test --plot-path plots_EEp_cor --do-correction-factor -i samples_processed_nonorm
python plot_weight_dist.py -d EEm --dataset test --plot-path plots_EEm_w -i samples_processed_nonorm
python plot_weight_dist.py -d EEm --dataset test --plot-path plots_EEm_cor --do-correction-factor -i samples_processed_nonorm


export yvar="probe_eta"
declare -a xvars_EB=("probe_pt" "fixedGridRhoAll" "probe_sieie" "probe_sieip" "probe_s4" "probe_r9" "probe_trkSumPtHollowConeDR03" "probe_trkSumPtSolidConeDR04" "probe_pfChargedIso" "probe_pfChargedIsoWorstVtx" "probe_ecalPFClusterIso" "probe_hcalPFClusterIso" "probe_etaWidth" "probe_phiWidth")
declare -a xvars_EE=("probe_pt" "fixedGridRhoAll" "probe_sieie" "probe_sieip" "probe_s4" "probe_r9" "probe_trkSumPtHollowConeDR03" "probe_trkSumPtSolidConeDR04" "probe_pfChargedIso" "probe_pfChargedIsoWorstVtx" "probe_ecalPFClusterIso" "probe_hcalPFClusterIso" "probe_etaWidth" "probe_phiWidth" "probe_esEnergyOverRawE" "probe_esEffSigmaRR")

for i in "${xvars_EB[@]}"; do python plot_2d_hist.py -d EB --dataset test --yvar ${yvar} --xvar $i --plot-path plots_2D_EB -i samples_processed_nonorm;done;
for i in "${xvars_EE[@]}"; do python plot_2d_hist.py -d EEm --dataset test --yvar ${yvar} --xvar $i --plot-path plots_2D_EEm -i samples_processed_nonorm;done;
for i in "${xvars_EE[@]}"; do python plot_2d_hist.py -d EEp --dataset test --yvar ${yvar} --xvar $i --plot-path plots_2D_EEp -i samples_processed_nonorm;done;

for i in "${xvars_EB[@]}"; do python plot_2d_weight.py -d EB --dataset test --yvar $i --plot-path plots_2D_EB_weight -i samples_processed_nonorm;done;
for i in "${xvars_EE[@]}"; do python plot_2d_weight.py -d EEm --dataset test --yvar $i  --plot-path plots_2D_EEm_weight -i samples_processed_nonorm;done;
for i in "${xvars_EE[@]}"; do python plot_2d_weight.py -d EEp --dataset test --yvar $i --plot-path plots_2D_EEp_weight -i samples_processed_nonorm;done

python plot_2d_weight.py -d EB --dataset test --yvar "weight" --plot-path plots_2D_EB_weight_vs_S2w -i samples_processed_nonorm
python plot_2d_weight.py -d EEm --dataset test --yvar "weight" --plot-path plots_2D_EEm_weight_vs_S2w -i samples_processed_nonorm
python plot_2d_weight.py -d EEp --dataset test --yvar "weight" --plot-path plots_2D_EEp_weight_vs_S2w -i samples_processed_nonorm