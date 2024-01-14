import torch
import torch.nn as nn
from transformers import RobertaModel, AutoTokenizer, RobertaConfig

class PhoBert_base (nn.Module):
    def __init__(self, freeze_backbone: bool, num_labels: int, drop_out: float):

        super(PhoBert_base,self).__init__()

        phobert_config = RobertaConfig.from_pretrained("vinai/phobert-base-v2")
        self.phobert = RobertaModel(config = phobert_config)
        self.phobert = RobertaModel.from_pretrained("vinai/phobert-base-v2")

        self.classifier = nn.Sequential(
                nn.Linear(768, 768),
                nn.Dropout(drop_out),
                nn.Linear(768, 512),  
                nn.ReLU(),           
                nn.Dropout(drop_out),
                nn.Linear(512, 256),   
                nn.ReLU(),           
                nn.Dropout(drop_out),
                nn.Linear(256, num_labels)
            )
        if freeze_backbone:
            for param in self.phobert.parameters():
                param.require_grad = False
        
    def forward(self, input_ids, attention_mask):
        feature = self.phobert(input_ids = input_ids, attention_mask = attention_mask)
        last_hidden = feature[0][:, 0, :]
        logits = self.classifier(last_hidden)

        return logits

