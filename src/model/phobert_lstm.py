import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class PhoBERTLSTM(nn.Module):
    def __init__(self, drop_out: float , num_labels: int):
        super(PhoBERTLSTM, self).__init__()
        phobert_config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
        self.phobert = AutoModel(config=phobert_config)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

        self.lstm = nn.LSTM(input_size=768, 
                            hidden_size=768,
                            batch_first=True, 
                            bidirectional=True, 
                            num_layers=1)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(768 * 2, num_labels)
        self.act = nn.LogSoftmax(dim=1)


    def forward(self, input_ids, attention_mask):

        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)[0]
        out, (h, c) = self.lstm(out)
        hidden = torch.cat((h[0], h[1]), dim=1)
        out = self.linear(self.dropout(hidden))

        return out
    