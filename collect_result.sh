#!/bin/bash
model_dir=$1
output_dir=$2
model_type=$3

#5 12 14 18 22 26 27 32 36 39 48 63 67 72 74 87 89 93 94 98
seed_list=(5 12 14 18 22 26 27 32 36 39)
metrics=(1 5 10 20 50 mrr)

DENSE_UNITS_ARR=(16 32 64 96)
RNN_UNITS_ARR=(16 32)

mkdir -p $output_dir

# Initialize the matrix
declare -A matrix
num_rows=$[${#metrics[@]}*${#seed_list[@]} + 1]
num_cols=$[${#DENSE_UNITS_ARR[@]}*${#RNN_UNITS_ARR[@]} + 1]

matrix[1,1]=$model_type

col_idx=1
for ((i=2;i<=num_rows;i++)) do
    matrix[$i,$col_idx]=${metrics[($i-2)%6]}
done

# Collect results
for rnn_unit in ${RNN_UNITS_ARR[@]}; do
  for dense_unit in ${DENSE_UNITS_ARR[@]}; do
  	col_idx=$[$col_idx + 1]
  	matrix[1,$col_idx]=D"$dense_unit"_H"$rnn_unit"
  	row_idx=2
    for seed_val in ${seed_list[@]}; do
    	file_path=$model_dir"/"$model_type"/D"$dense_unit"_H"$rnn_unit"/Seed_"$seed_val"/topN/out.txt"
		while read -r line; do
			v="$(cut -d',' -f2 <<<"$line")"
			matrix[$row_idx,$col_idx]=$v
			#echo $row_idx" "$col_idx
			#echo ${matrix[$row_idx,$col_idx]}
			row_idx=$[$row_idx + 1]
		done < $file_path
    done
  done
done
#declare -p matrix

# Print to file
output_file=$output_dir"/"$model_type".csv"
for ((i=1;i<=num_rows;i++)); do
	print_row=''
    for ((j=1;j<=num_cols;j++)); do
        print_row=$print_row""${matrix[$i,$j]}
        if [ $j -lt $num_cols ]; then
        	print_row=$print_row","
        fi
    done
    echo $print_row >> $output_file
done 
