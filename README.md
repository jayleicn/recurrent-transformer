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



## Resources
- Related works: [TVC (Video+Dialog Captioning)](https://github.com/jayleicn/TVCaption)


## Getting started
Coming soon! 

### Prerequisites
0. Clone this repository

1. Prepare feature files

2. Install dependencies.


### Training and Inference
We give examples on how to perform training and inference with MART.

1. MART training

2. MART inference


### Evaluation


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
This code is partly inspired by the following projects: 
[transformers](https://github.com/huggingface/transformers), 
[transformer-xl](https://github.com/kimiyoung/transformer-xl), 
[densecap](https://github.com/salesforce/densecap).

## Contact
jielei [at] cs.unc.edu
