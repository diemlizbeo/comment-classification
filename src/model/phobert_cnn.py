import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

class PhoBERTCNN_base(nn.Module):
    def __init__(self, drop_out: float, num_labels: int ):
        super(PhoBERTCNN_base, self).__init__()
        phobert_config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
        self.phobert = AutoModel(config=phobert_config)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

        self.conv1d = nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=3)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(1024, num_labels)

    def forward(self, input_ids, attention_mask):

        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)[0]
        out = self.conv1d(out.permute(0, 2, 1))
        out = F.max_pool1d(out, kernel_size=out.size(2)).squeeze(2)
        out = self.linear(self.dropout(out))
        
        return out
    