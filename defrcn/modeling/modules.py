import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from defrcn.utils.class_embedding import get_class_embed

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
    def __init__(self, input_size, output_size, class_names, embed_model='glove') -> None:
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.class_names = class_names
        self.embed_model = embed_model
        if embed_model == "glove":
            self.semantic_dim = 300
        elif embed_model == "clip":
            self.semantic_dim = 512 
        self.visual_dim = output_size
        
        self.class_embed = get_class_embed(self.class_names, self.embed_model)
        
        self.bg_embed_init = torch.randn(1, self.semantic_dim)
        self.bg_embed = torch.nn.parameter.Parameter(self.bg_embed_init.clone(), requires_grad=True)
        
        self.embed2vis_proj = nn.Linear(self.semantic_dim, self.visual_dim)
        self.combined2out_proj = nn.Linear(self.output_size * 2, self.output_size)
        
    def forward(self, feature_pooled, gt_classes):
        raise NotImplementedError()
        
        
class ConcatAddition(Addition):
    def __init__(self, input_size, output_size, class_names) -> None:
        super().__init__(input_size, output_size, class_names)
        
    def forward(self, feature_pooled, gt_classes):
        class_embed = torch.cat([self.class_embed, self.bg_embed])
        class_embed = self.embed2vis_proj(class_embed)
        semantic_features = class_embed[gt_classes]
        
        out = torch.cat((semantic_features, feature_pooled), dim=-1)
        out = self.combined2out_proj(out)

        return out
        
        
class AttentionAddition(Addition):
    def __init__(self, input_size, output_size, class_names, att_head_num=1) -> None:
        super().__init__(input_size, output_size, class_names)
        
        self.att_head_num = att_head_num
        
        self.attention = SingleSiameseAttention(d_model=self.input_size, output_size=self.output_size)
        self.init_scale = 0.02
        with torch.no_grad():
            self._init_parameters(self.attention, self.init_scale)
        
    def _init_parameters(self, module, init_scale):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=init_scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=init_scale)
            
    def forward(self, feature_pooled, gt_classes):
        class_embed = torch.cat([self.class_embed, self.bg_embed])
        class_embed = self.embed2vis_proj(class_embed)
        semantic_features = class_embed[gt_classes]
        
        visual_features = feature_pooled
        
        combined_feature = torch.cat([semantic_features, visual_features], dim=-1)
        combined_feature = self.combined2out_proj(combined_feature)
        
        visual_features = F.relu(visual_features)
        semantic_features= F.relu(semantic_features)
        
        out = self.attention(
            q=visual_features[None, :], 
            k=semantic_features[None, :], 
            v=combined_feature[None, :]
        )[0]
        
        out = F.relu(out)
        
        return out
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, att_dropout=0.1) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(att_dropout)
    
    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_model)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.bmm(attn, v)
        
        return out
        
        
class FFN(nn.Module):
    def __init__(self, d_model, output_size, dropout=0.0) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.dropout_ratio = dropout
        self.output_size = output_size
        
        self.ffn = nn.Sequential(   
            nn.Linear(self.d_model, self.output_size),
            nn.ReLU(), 
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.output_size, self.d_model),
            nn.Dropout(self.dropout_ratio)
        )
        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x):
        x = self.ffn(x) + self.norm(x) # Feed forward -> Add & Norm
        return x
    
    
class SingleSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. 
    No proj weights for V."""
    
    def __init__(self, d_model, output_size, n_head=1, dropout=0.0) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.output_size = output_size
        self.n_head = n_head
        self.dropout = dropout
        self.qk_w = nn.Linear(self.d_model, self.d_model * self.n_head, bias=False)
        self.attention = ScaledDotProductAttention(self.d_model, self.dropout)
        nn.init.normal_(
            self.qk_w.weight,
            mean=0.0, 
            std=np.sqrt(2.0 / (self.d_model + self.d_model))
            )
        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)
    
        self.linear1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), 
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), 
            nn.ReLU(inplace=True)
            )
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)
        
        self.ffn = FFN(self.d_model, self.output_size, self.dropout)
        
    def forward(self, q, k, v):
        # print(q.shape, k.shape, v.shape)
        b_sz, q_len, _ = q.size()
        b_sz, k_len, _ = k.size()
        b_sz, v_len, _ = v.size()
        
        q_residual = q
        q = self.qk_w(q).view(b_sz, q_len, self.n_head, self.d_model)
        k = self.qk_w(k).view(b_sz, k_len, self.n_head, self.d_model)
        v = v.view(b_sz, v_len, self.n_head, self.d_model)
        
        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(b_sz, -1, self.n_head, -1)
        dummy_v = torch.zeros(b_sz, 1, self.n_head, self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len + 1, self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len + 1, self.d_model)  # (n_head * b) x lv x d_model
      
        output = self.attention(q, k, v)

        output = output.view(self.n_head, b_sz, q_len, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(b_sz, q_len, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * q_residual)
        output2 = self.linear2(q_residual - output)
        output = self.linear3(torch.cat([output1, output2, q_residual], dim=2))
        output = self.ffn(output)

        return output
        