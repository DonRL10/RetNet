import os

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model"
TOKENISER_BIN = "tokenizer.bin"

class Tokeniser:
    def __init__(self):
        model_path = TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_path)

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.piece_size()
    
    def encode(self, s, bos, eos):
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t += [self.eos_id]
        return t
    
    def decode(self, t):
        return self.sp_model.decode(t)

    def export(self):
        tokens = []
        for i in range(self.n_words):
            t = self.sp_model.id_to_piece(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            elif len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                t = chr(int(t[3: 5], 16))
            t = t.replace('_', ' ')

            tokens.append(t)
        with open(TOKENISER_BIN, "wb") as f:
            for token in tokens:
                bytes = token.encode("utf8")
                f.write(len(bytes).to_bytes(4, 'little'))
                f.write(bytes)
       
    
if __name__ == "__main__":
    tok = Tokeniser()
    tok.export()
