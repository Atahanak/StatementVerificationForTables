import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import BertForSequenceClassification

class tabver(BertForSequenceClassification):
    def __init__(self):
        """    """
        super().__init__()


class ModelDefine(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super(ModelDefine, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        dim = 768
        self.classification_liner = nn.Linear(dim, 2)
        self.drop_out = nn.Dropout(0.1)
        self.pad_token = 0

    def forward(self, w_idxs1, type_idxs, mask_idxs=None, vm_low=None, vm_upper=None):

        embedding_output = self.bert.embeddings(w_idxs1, type_idxs * (0 if self.pad_token else 1))

        extended_attention_mask_base = mask_idxs.long().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = vm_low.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_low = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = vm_upper.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_upper = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_baseline = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = embedding_output
        for ly_idx, layer_module in enumerate(self.bert.encoder.layer):
            if ly_idx < 6:
                extended_attention_mask = extended_attention_mask_low
            else:
                extended_attention_mask = extended_attention_mask_upper

            hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
        last_layer = hidden_states

        max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2), kernel_size=last_layer.size(1)).squeeze(-1)
        max_pooling_fts = self.drop_out(max_pooling_fts)
        return self.classification_liner(max_pooling_fts)
