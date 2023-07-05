import re
import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            args,
    ):
        self.args=args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.word_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('data/reddit-cleanjokes.csv')

        word_string = ''
        for i in range(0, train_df.shape[0]):
            this_joke = train_df['Joke'][i]
            if 'http' in this_joke:
                continue
            word_string += re.sub("[^a-zA-Z0-9' ]", '', this_joke) + ' '
        return word_string.split(' ')
    
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def __len__(self):
        return len(self.word_indexes) - self.args.sequence_length
    
    def __getitem__(self, index):
        return(
            torch.tensor(self.word_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.word_indexes[index+1:index+self.args.sequence_length+1])
        )