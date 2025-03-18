import torch
import torch.nn as nn
import torch.nn.functional as F
from ..bert_model.bert_module import BertLMPredictionHead, BertLayer_Cross
from easydict import EasyDict as EDict

class TemporalSampling(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(100, width))
        self.bert_config = EDict(
            num_attention_heads=8,
            hidden_size=width,
            attention_head_size=width,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=width,
            vocab_size=1,
            num_layers=2
        )
        self.layer_ca = nn.ModuleList([BertLayer_Cross(self.bert_config) for _ in range(self.bert_config.num_layers)])
        self.head = BertLMPredictionHead(self.bert_config)

    def forward(self, x, query=None):
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze().unsqueeze(0)
        for i in range(self.bert_config.num_layers):
            x, _ = self.layer_ca[i](x, query)
        logits = self.head(x).squeeze()
        return logits


class SpatialActivation(nn.Module):
    def __init__(self, width, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(100, width))
        self.bert_config = EDict(
            num_attention_heads=8,
            hidden_size=width,
            attention_head_size=width,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=width,
            vocab_size=vocab_size,
            num_layers=2
        )
        self.layer_ca = nn.ModuleList([BertLayer_Cross(self.bert_config) for _ in range(self.bert_config.num_layers)])
        self.head = BertLMPredictionHead(self.bert_config)

    def forward(self, input, init_q=None):
        input = input.permute(0, 2, 3, 1)
        x = input.reshape(input.size(0), -1, 256)
        query = torch.zeros(x.size(0), 1, x.size(-1)).to(x.device) if init_q is None else init_q.repeat(x.size(0), 1, 1)
       
        for i in range(self.bert_config.num_layers):
            query, att_map = self.layer_ca[i](query, x)
        att_map = att_map.sum(1).squeeze(1).sigmoid()
        att_map = (att_map - att_map.min(dim=1, keepdim=True)[0]) / (att_map.max(dim=1, keepdim=True)[0] - att_map.min(dim=1, keepdim=True)[0])
        
        logits = self.head(query).mean(0)
        return logits, att_map