
import torch
from torch import nn

def phobert_collator(batch):

    input_ids_list, attn_mask_list, label_list = [], [], []

    for input_ids, attn_mask, label in batch:
        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
        label_list.append(label)
    label_list = torch.tensor(label_list)
    
    input_ids_list = nn.utils.rnn.pad_sequence(input_ids_list)
    attn_mask_list = nn.utils.rnn.pad_sequence(attn_mask_list)

    input_ids_list = torch.permute(input_ids_list, (1, 0))
    attn_mask_list = torch.permute(attn_mask_list, (1, 0))
    
    return input_ids_list, attn_mask_list, label_list


def fasttext_collator(batch):

    vectors_list, label_list = [], []
    for text, label in batch:
        vectors_list.append(text)
        label_list.append(label)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    vectors_list = nn.utils.rnn.pad_sequence(vectors_list)

    return vectors_list, label_list

