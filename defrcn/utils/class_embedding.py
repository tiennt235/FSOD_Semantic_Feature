import numpy as np
import torch

def get_class_embed(class_names, model, include_bg=False):
    with torch.no_grad():
        semantic_features = []
        for class_name in class_names:
            semantic_features.append(np.loadtxt(f"datasets/{model}/{class_name}.txt"))
        if include_bg:
            semantic_features.append(np.loadtxt(f"datasets/{model}/background.txt"))
        semantic_features = torch.tensor(np.array(semantic_features))
            
    return semantic_features.to('cuda').type(torch.float)