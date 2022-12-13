
CUDA_VISIBLE_DEVICES=0 python3 test_all.py \
 --dataset HistoSR --gpu 0 \
--model_name 12_13_13_3 --episodes best \
--use_tpM --save_images > ./logs/test_results.txt
