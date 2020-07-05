#!/usr/bin/env bash
# Usage:
# $ bash scripts/download_pretrained.sh

# download and untar vocab files, yoou should see a dir named `cache` under ${PROJECT_ROOT}
wget http://vision.cs.unc.edu/jielei/project/mart_data/mart_cache.tar.gz
tar -xf mart_cache.tar.gz
rm -rf mart_cache.tar.gz
# download pre-trained MART model, you should see
# a dir named `anet_re_init_2019_10_01_11_34_22` under ${PROJECT_ROOT}
mkdir -p results && cd results || exit
wget http://vision.cs.unc.edu/jielei/project/mart_data/mart_anet_re_init_2019_10_01_11_34_22.tar.gz
tar -xf mart_anet_re_init_2019_10_01_11_34_22.tar.gz
rm -rf mart_anet_re_init_2019_10_01_11_34_22.tar.gz
