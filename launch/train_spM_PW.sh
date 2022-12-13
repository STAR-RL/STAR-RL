
echo 'Train spatial manager and patch worker together.'
echo 'Logging information are saved at ./outputs/spM_PW_log.txt.'
CUDA_VISIBLE_DEVICES=0 python3 train_spM_PW.py --gpu 0 --dataset HistoSR  --output_name spM_PW_log.txt