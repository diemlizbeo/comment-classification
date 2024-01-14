import os
import re
import yaml
import time
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyvi import ViTokenizer


fasttext_embedding_path = "./src/embedding/fasttext_tiki_train_dev.model"

if __name__ == "__main__":
    print(os.getcwd())
    train_df =  pd.read_table(os.path.join(
            './data/train/', "sents.txt"), names=['sents'])
    dev_df = pd.read_table(os.path.join(
            './data/dev/', "sents.txt"), names=['sents'])

    sents = list(np.concatenate(np.vstack((train_df.values, dev_df.values))))
    print(f"Total sentences: {len(sents)}")

    x_tokenized = []
    for sent in tqdm(sents, leave=True, desc="FastText training"):
        sent = re.sub(r'[^\w\s]', '', str(sent))
        tokens = ViTokenizer.tokenize(sent)
        tokens = tokens.split()
        for token in tokens:
            for t in token:
                if t.isnumeric() or t.isdigit():
                    tokens.remove(token)
                    break
        x_tokenized.append(tokens)

    start = time.time()
    model = gensim.models.FastText(sentences=x_tokenized, vector_size=300, window=3, min_count=5, workers=4)
    model.build_vocab(corpus_iterable=x_tokenized)
    model.train(corpus_iterable=x_tokenized, total_examples=len(x_tokenized), epochs=15)
    end = time.time()
    total_time = end - start
    
    print("Training finish!")
    print(f"FastText training total time is: {total_time:.2f} seconds")

    save_dir = "/".join(fasttext_embedding_path.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(fasttext_embedding_path)
