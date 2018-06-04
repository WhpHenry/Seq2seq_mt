# [Seq2seq English-French Translation]

Code is based on:
 [NELSONZHAO/machine_translation_seq2seq](https://github.com/NELSONZHAO/zhihu/tree/master/machine_translation_seq2seq) 

## Basic Enviornment and Requirement:
- Python 3.6
- Tensorflow-gpu 1.4.0

### Train self-model:
```
python run.py <train> 
```
data files' path can be modified in utils.py 

### Use Trained Models:
```
python run.py predict 
```
then input your english sentence

