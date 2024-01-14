import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class PhoBert_large(nn.Module):
    def __init__(self, drop_out: float, num_labels: int):
        super(PhoBert_large, self).__init__()
        phobert_config = AutoConfig.from_pretrained("vinai/phobert-large")
        self.phobert = AutoModel(config=phobert_config)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-large")
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(drop_out),
            nn.Linear(768, 512),  
            nn.ReLU(),            
            nn.Dropout(drop_out),
            nn.Linear(512, num_labels))


    def forward(self, input_ids, attention_mask):

        feature = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = feature[0][:, 0, :]
        logits = self.classifier(last_hidden)

        return logits