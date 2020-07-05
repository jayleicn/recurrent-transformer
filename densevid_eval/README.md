# Dense Captioning Events in Video - Evaluation Code

This is a modified copy from https://github.com/jamespark3922/densevid_eval, 
which is again a modified copy of [densevid_eval](https://github.com/ranjaykrishna/densevid_eval).
Instead of using sentence metrics, we evaluate captions at the paragraph level, 
as described in [Move Forward and Tell (ECCV18)](https://arxiv.org/abs/1807.10018)

## Usage
```
python para-evaluate.py -s YOUR_SUBMISSION_FILE.JSON --verbose
```

## Paper
Visit [the project page](http://cs.stanford.edu/people/ranjaykrishna/densevid) for details on activitynet captions.

## Citation
```
@inproceedings{krishna2017dense,
    title={Dense-Captioning Events in Videos},
    author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
    booktitle={ArXiv},
    year={2017}
}
```

## License

MIT License copyright Ranjay Krishna

