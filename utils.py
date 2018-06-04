
_CODE = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
_PAD_i = 0
_UNK_i = 1
_GO_i  = 2
_EOS_i = 3

_PATH = 'data/'
_SOURCE = 'small_vocab_en'
_TARGET = 'small_vocab_fr'

def load_data(fs):
    with open(fs, 'r', encoding='utf-8') as f:
        return f.read()

def build_vocab_list(data, code):
    return code + list(set(data.lower().split())) 

def build_vocab_dict(vocab, ttype='txt2int'):
    # return {index:word} if ttype is txt2int 
    # else return {index:word}, means int2txt
    if ttype == 'txt2int':
        return {w:i for i, w in enumerate(vocab)}
    return {i:w for i, w in enumerate(vocab)}

def seq_transform(seq, vocab_dict):
    if isinstance(seq, str):
        seq = seq.lower().split()
    return [vocab_dict.get(s, _UNK_i) for s in seq]

def padding(seq, max_len=20, do_decode=False):
    # default seq is num sequence ,after transformed
    # if do_decode, seq should be decoder input, 
    # which need add <EOS> as the end of decode 
    res = seq[:]
    gap = max_len - len(res)
    if gap < 0:
        res = res[:max_len]
    else:
        res += [_PAD_i] * gap
    if do_decode:
        res[-1] = _EOS_i
    return res


source = load_data(_PATH + _SOURCE)
target = load_data(_PATH + _TARGET)
source_vob = build_vocab_list(source, _CODE[:2]) 
target_vob = build_vocab_list(target, _CODE)

s_txt2int_dict =  build_vocab_dict(source_vob)
t_txt2int_dict =  build_vocab_dict(target_vob)

s_int2txt_dict = build_vocab_dict(source_vob, 'int2txt')
t_int2txt_dict = build_vocab_dict(target_vob, 'int2txt')

# test

def test():
    sseq = source.split('\n')[0]
    tseq = target.split('\n')[0]

    sseqi = seq_transform(sseq, s_txt2int_dict)
    tseqi = seq_transform(tseq, t_txt2int_dict)

    pseeqi = padding(sseqi)
    ptseqi = padding(tseqi, do_decode=True)
    print('source: ', sseq, '\n', sseqi, '\n', pseeqi)
    print('target: ', tseq, '\n', tseqi, '\n', ptseqi)

    print('recover source: ', ' '.join(seq_transform(pseeqi, s_int2txt_dict)))
    print('recover target: ', ' '.join(seq_transform(ptseqi, t_int2txt_dict)))

# test end

