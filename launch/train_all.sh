echo 'Train spatial manager, temporal manager and patch worker together'
echo 'Logging information are saved at ./outputs/spM_tpM_PW_log.txt.'
CUDA_VISIBLE_DEVICES=0 python3 train_all.py --gpu 0 --dataset HistoSR  --output_name spM_tpM_PW_log.txt
