#!/usr/bin/env bash
# Usage:
# $ bash scripts/build_vocab.sh anet

dset_name=$1  # [anet, yc2]

echo "---------------------------------------------------------"
echo ">>>>>>>> Running on ${dset_name} Dataset"
if [[ ${dset_name} == "anet" ]]; then
    min_word_count=5
    train_path="./densevid_eval/anet_data/train.json"
elif [[ ${dset_name} == "yc2" ]]; then
    min_word_count=3
    train_path="./densevid_eval/yc2_data/yc2_train_anet_format.json"
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

# download and extract http://nlp.stanford.edu/data/glove.6B.zip,
# modify glove_path to your downloaded glove path
glove_path="path/to/glove.6B.300d.txt"

python src/build_vocab.py \
--train_path ${train_path} \
--dset_name ${dset_name} \
--cache ./cache \
--min_word_count ${min_word_count} \
--raw_glove_path ${glove_path}


