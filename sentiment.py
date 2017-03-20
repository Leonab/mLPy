import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lex(pos, neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


def handle(sample, lexicon, classification):
    featureset = []

    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l)
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeroes(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset

def create_sets_labels(pos, neg, test_size):
    lexicon = create_lex(pos,neg)
    features = []
    features += handle('pos.txt')
    features += handle('neg.txt')
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][:-testing_size])
    test_y = list(features[:,1][:-testing_size])

    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_sets_labels('pos.txt','neg.txt')
    with open('sentiment.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
