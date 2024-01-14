import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from .collator import fasttext_collator, phobert_collator
import unicodedata
from underthesea import word_tokenize, sent_tokenize
import re
import os
import torch
from torch.utils.data import Dataset
from transformers import PhobertTokenizer
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.utils.class_weight import compute_class_weight



class TextDataset(Dataset):
    def __init__(self, data_dir: str, model_type: str, fasttext_embedding: str = None):

       
        self.features = pd.read_table(os.path.join(data_dir, "sents.txt"), names=['sents'])
        self.labels = pd.read_table(os.path.join(data_dir, "sentiments.txt"), names=["labels"])

        self.model_type = model_type

        if self.model_type == 'fasttext':
            self.word_vec = KeyedVectors.load(fasttext_embedding)

        self.tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base-v2")

        y = self.labels.values

        self.class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(np.ravel(y)), y=np.ravel(y)), dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.features.iloc[index].values[0]
        y = self.labels.iloc[index].values[0]
        processed_txt = preprocess(X)

        if self.model_type == "phobert":
            tokens = self.tokenizer( processed_txt, truncation=True, padding=True, max_length=256)
            return torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"]), torch.tensor(y)
        else:
            x_embed = []
            for x_token in processed_txt:
                x_embed.append(torch.unsqueeze(torch.tensor(self.word_vec.wv[x_token]), dim=0))
            return torch.cat(x_embed, dim=0), torch.tensor(y)

class DataModule(LightningDataModule):
    def __init__(self, root_data_dir: str, model_type: str, batch_size: int, num_workers: int, fasttext_embedding: str = None):
        super().__init__()
        self.root_data_dir = root_data_dir
        print('MODEL TYPE:',model_type)
        if model_type == "phobert":
            self.collate_fn = phobert_collator
        else:
            self.collate_fn = fasttext_collator

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fasttext = fasttext_embedding
        if model_type == 'phobert':
            self.fasttext = None
        self.model_type = model_type

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = TextDataset(data_dir=os.path.join(self.root_data_dir, "train"), model_type=self.model_type, fasttext_embedding=self.fasttext)
            self.val_data = TextDataset(data_dir=os.path.join(self.root_data_dir, "dev"), model_type=self.model_type, fasttext_embedding=self.fasttext)

        if stage == "test":
            self.test_data = TextDataset(data_dir=os.path.join( self.root_data_dir, "test"), model_type=self.model_type, fasttext_embedding=self.fasttext)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)


def preprocess(text):

    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = " ".join([word_tokenize(sent, format='text') for sent in  sent_tokenize(text)])
    return text