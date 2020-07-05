#!/usr/bin/env bash
# Usage:
# $ bash scripts/eval.sh anet val /path/to/greedy_pred_val.json
# The generated metrics will be save at /path/to/greedy_pred_val_combined_metrics.json

dset_name=$1  # [anet, yc2]
split_name=$2  # [val, test] for anet, val for yc2
filename=$3

echo "---------------------------------------------------------"
echo ">>>>>>>> Running evaluation on ${dset_name} dataset ${split_name} split, with file ${filename}"
if [[ ${dset_name} == "anet" ]]; then
    ref_paths=()
    ref_paths+=(./densevid_eval/anet_data/anet_entities_${split_name}_1_para.json)
    ref_paths+=(./densevid_eval/anet_data/anet_entities_${split_name}_2_para.json)
elif [[ ${dset_name} == "yc2" ]]; then
    ref_paths=()
    ref_paths+=(./densevid_eval/yc2_data/yc2_val_anet_format_para.json)
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

# requires java
echo "---------------coco language "
python densevid_eval/para-evaluate.py \
-s "${filename}" \
-r "${ref_paths[@]}" \
-o "${filename%%.*}_lang_metrics.json" \
-v

echo "---------------repetition"
python densevid_eval/evaluateRepetition.py \
-s "${filename}" \
-r "${ref_paths[0]}" \
-o "${filename%%.*}_rep_metrics.json"
echo "save at ${filename%%.*}_rep_metrics.json"

echo "---------------basic stat"
python densevid_eval/get_caption_stat.py \
-s "${filename}" \
-r "${ref_paths[0]}" \
-o "${filename%%.*}_stat_metrics.json"

echo "---------------Combine all the metrics in a single file"
python densevid_eval/merge_dicts_by_prefix.py \
-t "${filename%%.*}_*_metrics.json" \
-o "${filename%%.*}_combined_metrics.json"

