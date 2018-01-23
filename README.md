# FusionNet

This is my pytorch implementation of the [FusionNet: Fusing via Fully-aware Attention with Application to Machine Comprehension](https://openreview.net/pdf?id=BJIgi_eCZ) for chinese machine comprehension dataset (not public dataset). However, I am now implementing the preprocessing/decoding for SQuAD, I think the SQuAD compatiable version will be released soon.

Feel free to use/modify it. Any improvement will be appreciated.

## HOW TO RUN:
```
$ python3 preprocess.py
$ python3 train.py
```

## TODO:
I will implement following word feature soon
- <strike>Glove word vector and fine-tune (Maybe fasttext?) </strike>
- CoVe vector
- POS, NER, normalized term frequency 
- <strike>context word appears in the quesiton?</strike>

## Result:
[Kaggle competition](https://www.kaggle.com/c/ml-2017fall-final-chinese-qa/leaderboard)

## References:
- [FUSION NET: FUSING VIA FULLY-AWARE ATTENTION WITH APPLICATION TO MACHINE COMPREHENSION](https://openreview.net/pdf?id=BJIgi_eCZ)
- [Awesome implementation of the DrQA](https://github.com/facebookresearch/DrQA)