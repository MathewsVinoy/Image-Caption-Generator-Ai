import torch

class DataConverter:
    words_to_num = {char: idx for idx, char in enumerate(set(tokens))}
    num_to_words = {idx: char for idx, char in enumerate(set(tokens))}

    encode = lambda s: [words_to_num[c] for c in word_tokenize(s)]
    def encode2(l):

        l=str(l)
        value = [words_to_num[c] for c in word_tokenize(l)]
        while len(value) <82:
            value.append(0)
        return torch.tensor(value, dtype=torch.float32)

        
    def decoder(l):
        return ' '.join([num_to_words.get(i,'<UNK>') for i in l])