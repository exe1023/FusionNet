# FusionNet

My implementation of the [FusionNet](https://openreview.net/pdf?id=BJIgi_eCZ) for chinese machine comprehension dataset (not public dataset).

I think it is easy to modify `preprocess.py` to use this model on SQuAD dataset.

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