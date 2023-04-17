import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from ..utils.get_semantic_features import get_semantic_features

class MLPAdapter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(MLPAdapter, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size , output_size),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.model(x)
        return out


class Addition(nn.Module):
    def __init__(self, output_size, class_names, semantic_dim=300) -> None:
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.output_size = output_size
        self.class_names = class_names
        
        self.semantic_adapter = nn.Linear(self.semantic_dim, self.output_size)
        self.sem_vis_adapter = nn.Linear(self.output_size + self.output_size, self.output_size)
        self.semantic_features = get_semantic_features(self.class_names, model="glove").cuda()
        self.bg_feature_init = torch.randn(1, self.semantic_dim)
        self.bg_feature = nn.parameter.Parameter(self.bg_feature_init.clone(), requires_grad=True)
        
    def forward(self, feature_pooled, gt_classes):
        raise NotImplementedError()
        
class ConcatAddition(Addition):
    def __init__(self, output_size, class_names) -> None:
        super().__init__(output_size, class_names)
    
    def forward(self, feature_pooled, gt_classes):
        semantic_features = torch.cat([self.semantic_features, self.bg_feature])
        semantic_features = self.semantic_adapter(semantic_features)
        semantic_features = semantic_features[gt_classes]
        
        out = torch.cat((semantic_features, feature_pooled), dim=-1)
        out = self.sem_vis_adapter(out)

        return out
        
        
class AttentionAddition(Addition):
    def __init__(self, output_size, semantic_dim=300, att_head_num=1) -> None:
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.output_size = output_size

        self.semantic_adapter = nn.Linear(self.semantic_dim, self.output_size)
        self.sem_vis_adapter = nn.Linear(self.output_size + self.output_size, self.output_size)
        
        self.att_head_num = att_head_num
        # self.w_qk = nn.Linear(self.output_size, self.output_size)
        # self.attention = ScaledDotProductAttention(d_model=output_size)
    
    def forward(self, feature_pooled, gt_classes):
        text_embed = self._get_text_embed(gt_classes)
        lv_feat = self.sem_vis_adapter(torch.cat((text_embed, feature_pooled), dim=-1))
        
        # change to (1, size, size)
        feature_pooled = text_embed[None, :]
        text_embed = text_embed[None, :]
        lv_feat = lv_feat[None, :]
        
        # print(feature_pooled.size())
        # print(text_embed.size())
        # print(lv_feat.size())
        
        batch_size, q_len, _ = feature_pooled.size()
        batch_size, k_len, _ = text_embed.size()
        batch_size, v_len, _ = lv_feat.size()
        
        q = self.w_qk(feature_pooled) #.view(batch_size, q_len, self.att_head_num, self.output_size)
        k = self.w_qk(text_embed) #.view(batch_size, k_len, self.att_head_num, self.output_size)
        v = lv_feat #.view(batch_size, v_len, self.att_head_num, self.output_size)
        
        out = self.attention(q, k, v)
        # out = self.l_v_adapter(lv_feat)
        out = torch.squeeze(out, dim=0)
        # print(out.shape)
        return out
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, att_dropout=0.1) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(att_dropout)
    
    def forward(self, q, k, v):
        # print(q.size())
        # print(k.size())
        # print(v.size())
        
        attn = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_model)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.bmm(attn, v)
        
        return out
        