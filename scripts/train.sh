#!/usr/bin/env bash
# Usage:
#   $ bash {SCRIPT.sh} {DATASET} [Any flags available in train.py, could also be empty]
#   DATASET: `anet` or `yc2`
#   Note the additional flags added will overwrite the specified flags below,
#   i.e., if `--exp_id run1` is specified, it will overwrite `--exp_id init` below.
# Examples:
#   anet debug mode: $ bash scripts/train.sh anet -debug
#   yc2 training mode: $ bash scripts/train.sh yc2

dset_name=$1  # [anet, yc2]
model_type=$2  # [mart, xl, xlrg, mtrans, mart_no_recurrence]

data_dir="./densevid_eval/${dset_name}_data"
v_feat_dir="./video_feature/${dset_name}_trainval"
dur_file="./video_feature/${dset_name}_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset"
if [[ ${dset_name} == "anet" ]]; then
    max_n_sen=6
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=100
elif [[ ${dset_name} == "yc2" ]]; then
    max_n_sen=12
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=100
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

echo ">>>>>>>> Model type ${model_type}"
echo "---------------------------------------------------------"
extra_args=()
if [[ ${model_type} == "mart" ]]; then   # MART
    extra_args+=(--recurrent)
elif [[ ${dset_name} == "xl" ]]; then    # Transformer-XL
    extra_args+=(--recurrent)
    extra_args+=(--xl)
elif [[ ${dset_name} == "xlrg" ]]; then  # Transformer-XLRG
    extra_args+=(--recurrent)
    extra_args+=(--xl)
    extra_args+=(--xl_grad)
    extra_args+=(--n_epoch)
    extra_args+=(100)  # it takes longer for this model to converge
elif [[ ${dset_name} == "mtrans" ]]; then    # Vanilla Transformer (from Masked Transformer paper)
    extra_args+=(--mtrans)
elif [[ ${dset_name} == "mart_no_recurrence" ]]; then    # MART w/o recurrence
    extra_args=()
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

python src/train.py \
--dset_name ${dset_name} \
--data_dir ${data_dir} \
--video_feature_dir ${v_feat_dir} \
--v_duration_file ${dur_file} \
--word2idx_path ${word2idx_path} \
--glove_path ${glove_path} \
--max_n_sen ${max_n_sen} \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
--exp_id init \
${extra_args[@]} \
${@:3}
