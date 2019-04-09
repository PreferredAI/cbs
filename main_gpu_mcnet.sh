#!/bin/bash
source ~/conda3/bin/activate tensorflow_gpu
python --version

data_dir=$1
output_dir=$2
model_type=$3
device_id=$4

#5 12 14 18 22 26 27 32 36 39 48 63 67 72 74 87 89 93 94 98
seed_list=(5 12 14 18 22 26 27 32 36 39)

DENSE_UNITS_ARR=(16 32 64 96)

for dense_unit in ${DENSE_UNITS_ARR[@]}; do
  sub_dir=$output_dir"/"$model_type"/D"$dense_unit
  mkdir -p $sub_dir
  echo $sub_dir"=>"$device_id
  if [[ $sub_dir != *"H64"* ]]; then
    for seed_val in ${seed_list[@]}; do python3 -u main_gpu.py --device_id $device_id --data_dir $data_dir --output_dir $sub_dir/"Seed_"$seed_val --seed $seed_val --dense_unit $dense_unit --model_type $model_type --top_k 10 --nb_epoch 20 --train_mode --prediction_mode; done > $sub_dir"/all_log.txt" 2>&1 &
  else
    for seed_val in ${seed_list[@]}; do python3 -u main_gpu.py --device_id $device_id --data_dir $data_dir --output_dir $sub_dir/"Seed_"$seed_val --seed $seed_val --dense_unit $dense_unit --model_type $model_type --top_k 10 --nb_epoch 20 --tensorboard_dir $sub_dir/"Seed_"$seed_val"/tensorboard"  --train_mode --prediction_mode; done > $sub_dir"/all_log.txt" 2>&1 &
  fi
    
  device_id=$[$device_id+1]
  if((device_id>7)); then
    device_id=$4
  fi      
done
