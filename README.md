MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning
=====
PyTorch code for our ACL 2020 paper ["MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning"](https://arxiv.org/abs/2005.05402)
by [Jie Lei](http://www.cs.unc.edu/~jielei/), [Liwei Wang](http://www.deepcv.net/),
[Yelong Shen](https://scholar.google.com/citations?user=S6OFEFEAAAAJ&hl=en), 
[Dong Yu](https://sites.google.com/site/dongyu888/),
[Tamara L. Berg](http://tamaraberg.com/), and [Mohit Bansal](http://www.cs.unc.edu/~mbansal/)

Generating multi-sentence descriptions for videos is one of the most challenging captioning tasks 
due to its high requirements for not only visual relevance but also discourse-based coherence 
across the sentences in the paragraph. Towards this goal, we propose a new approach called 
Memory-Augmented Recurrent Transformer (MART), which uses a memory module to augment 
the transformer architecture. The memory module generates a highly summarized memory state 
from the video segments and the sentence history so as to help better prediction of the 
next sentence (w.r.t. coreference and repetition aspects), thus encouraging coherent 
paragraph generation. Extensive experiments, human evaluations, 
and qualitative analyses on two popular datasets ActivityNet Captions and YouCookII 
show that MART generates more coherent and less repetitive paragraph captions than baseline methods, 
while maintaining relevance to the input video events.



## Related works:
- [TVC (Video+Dialogue Captioning)](https://github.com/jayleicn/TVCaption). 
- [TVR (Video+Dialogue Retrieval)](https://github.com/jayleicn/TVRetrieval). 
- [TVQA (Localized Video QA)](https://github.com/jayleicn/TVQA). 
- [TVQA+ (Spatio-Temporal Video QA)](https://github.com/jayleicn/TVQAplus).

## Getting started
### Prerequisites
0. Clone this repository
```
# no need to add --recursive as all dependencies are copied into this repo.
git clone https://github.com/jayleicn/recurrent-transformer.git
cd recurrent-transformer
```

1. Prepare feature files

Download features from Google Drive: [rt_anet_feat.tar.gz (39GB)](https://drive.google.com/file/d/1mbTmMOFWcO30PIcuSpYiZ1rqoy5ltE3A/view?usp=sharing) 
and [rt_yc2_feat.tar.gz (12GB)](https://drive.google.com/file/d/1mj76DwNexFCYovUt8BREeHccQn_z_By9/view?usp=sharing).
These features are repacked from features provided by [densecap](https://github.com/salesforce/densecap#annotation-and-feature). 
```
mkdir video_feature && cd video_feature
tar -xf path/to/rt_anet_feat.tar.gz 
tar -xf path/to/rt_yc2_feat.tar.gz 
```

2. Install dependencies
- Python 2.7
- PyTorch 1.1.0
- nltk
- easydict
- tqdm
- tensorboardX

3. Add project root to `PYTHONPATH`
```
source setup.sh
```
Note that you need to do this each time you start a new session.


### Training and Inference
We give examples on how to perform training and inference with MART.

0. Build Vocabulary
```
bash scripts/build_vocab.sh DATASET_NAME
```
`DATASET_NAME` can be `anet` for ActivityNet Captions or `yc2` for YouCookII.


1. MART training

The general training command is:
```
bash scripts/train.sh DATASET_NAME MODEL_TYPE
```
`MODEL_TYPE` can be one of `[mart, xl, xlrg, mtrans, mart_no_recurrence]`, see details below.

| MODEL_TYPE         | Description                            |
|--------------------|----------------------------------------|
| mart               | Memory Augmented Recurrent Transformer |
| xl                 | Transformer-XL                         |
| xlrg               | Transformer-XL with recurrent gradient |
| mtrans             | Vanilla Transformer                    |
| mart_no_recurrence | mart with recurrence disabled          |


To train our MART model on ActivityNet Captions:
```
bash scripts/train.sh anet mart
```
Training log and model will be saved at `results/anet_re_*`.  
Once you have a trained model, you can follow the instructions below to generate captions. 


2. Generate captions 
```
bash scripts/translate_greedy.sh anet_re_* val
```
Replace `anet_re_*` with your own model directory name. 
The generated captions are saved at `results/anet_re_*/greedy_pred_val.json`


3. Evaluate generated captions
```
bash scripts/eval.sh anet val results/anet_re_*/greedy_pred_val.json
```
The results should be comparable with the results we present at Table 2 of the paper. 
E.g., B@4 10.33; R@4 5.18.

## Citations
If you find this code useful for your research, please cite our paper:
```
@inproceedings{lei2020mart,
  title={MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning},
  author={Lei, Jie and Wang, Liwei and Shen, Yelong and Yu, Dong and Berg, Tamara L and Bansal, Mohit},
  booktitle={ACL},
  year={2020}
}
```

## Others
This code used resources from the following projects: 
[transformers](https://github.com/huggingface/transformers), 
[transformer-xl](https://github.com/kimiyoung/transformer-xl), 
[densecap](https://github.com/salesforce/densecap),
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## Contact
jielei [at] cs.unc.edu
