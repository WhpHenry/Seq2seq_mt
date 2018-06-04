import sys

from model import Seq2seq
from utils import seq_transform, padding
from utils import source, target, s_txt2int_dict, t_txt2int_dict, t_int2txt_dict

s2s = Seq2seq(source_vocab_size = len(s_txt2int_dict), 
              target_vocab_size = len(t_txt2int_dict))
source_data = [seq_transform(seq, s_txt2int_dict) for seq in source.lower().split('\n')]
target_data = [seq_transform(seq, t_txt2int_dict) for seq in target.lower().split('\n')]

source_data = [padding(seq) for seq in source_data]
target_data = [padding(seq, do_decode=True) for seq in target_data]

def train(seq2seq_m):
    seq2seq_m.training(source_data, target_data, len(source_data))

def predict(seq2seq_m, sentence='i dislike grapefruit'):
    sentence_int_seq = seq_transform(sentence, s_txt2int_dict)
    logit = seq2seq_m.inference(sentence_int_seq)
    res = seq_transform(logit, t_int2txt_dict)
    print("model predict return: {}\ntrans to source symbols: {}".format(logit, ' '.join(res)))

def main(argv, seq2seq_m):
    if len(argv) <= 1 or argv[1] == 'train':
        train(seq2seq_m)
    else:
        sentence = input('please input english sentence here:\n')
        predict(seq2seq_m, sentence)


if __name__ == '__main__':
    main(sys.argv, s2s)
