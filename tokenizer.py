import sentencepiece as spm

class CharTokenizer :
    def __init__(self):
        self.char_to_id = {'<bos>': 0, '<eos>': 1, '<pad>': 2 }
        self.id_to_char = {0 : '<bos>', 1 : '<eos>', 2 : '<pad>'}

    def fit(self, texts):
        chars_uniques = set(''.join(texts))
        for idx, char in enumerate(sorted(chars_uniques)):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

    def encode(self, text):
        return [self.char_to_id['<bos>']] + [self.char_to_id[c] for c in text if c in self.char_to_id] + [self.char_to_id['<eos>']]

    def decode(self, ids):
        return ''.join([self.id_to_char[i] for i in ids if i > 2])



class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, out_type=int):

        return self.sp.encode(text, out_type= out_type)

    def decode(self, ids):
        return self.sp.decode(ids)
