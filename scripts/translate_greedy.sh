#!/usr/bin/env bash
res_dir=$1
split_name=$2
python src/translate.py \
--res_dir=results/${res_dir} \
--eval_splits=${split_name} \
${@:3}
