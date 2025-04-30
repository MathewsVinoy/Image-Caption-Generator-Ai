import torch                                # type: ignore
from nltk.tokenize import word_tokenize     # type: ignore

class DataConverter:
    def __init__(self,tokens):
        self.words_to_num = {char: idx for idx, char in enumerate(set(tokens))}
        self.num_to_words = {idx: char for idx, char in enumerate(set(tokens))}

    def encode2(self,l):
        l=str(l)
        value = [self.words_to_num[c] for c in word_tokenize(l)]
        while len(value) <82:
            value.append(0)
        return torch.tensor(value, dtype=torch.float32)

        
    def decoder(self,l):
        return ' '.join([self.num_to_words.get(i,'<UNK>') for i in l])