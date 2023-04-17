import numpy as np
import torch

def get_semantic_features(class_names, model, semantic_dim=300):
    with torch.no_grad():
        semantic_features = []
        for class_name in class_names:
            semantic_features.append(np.loadtxt(f"datasets/{model}/{class_name}.txt"))
        semantic_features = torch.tensor(np.array(semantic_features))
            
    return semantic_features.to('cuda').type(torch.float)