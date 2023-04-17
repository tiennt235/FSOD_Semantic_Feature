import numpy as np
import torch
import clip
from pathlib import Path
import argparse
from torchnlp.word_to_vector import GloVe

VOC_CATEGORIES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_GLOVE_MAP = {
    "aeroplane": "aeroplane",
    "bicycle": "bicycle",
    "boat": "boat",
    "bottle": "bottle",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "train": "train",
    "tvmonitor": "tv",
    "bird": "bird",
    "bus": "bus",
    "cow": "cow",
    "motorbike": "motorbike",
    "sofa": "sofa"
    }
COCO_CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
COCO_GLOVE_MAP = {
    "person" : "person",
    "bicycle" : "bicycle",
    "car" : "car",
    "motorcycle" : "motorcycle",
    "airplane" : "airplane",
    "bus" : "bus",
    "train" : "train",
    "truck" : "truck",
    "boat" : "boat",
    "traffic light" : "stoplight",
    "fire hydrant" : "hydrant",
    "stop sign" : "stop sign",
    "parking meter" : "parking meter",
    "bench" : "bench",
    "bird" : "bird",
    "cat" : "cat",
    "dog" : "dog",
    "horse" : "horse",
    "sheep" : "sheep",
    "cow" : "cow",
    "elephant" : "elephant",
    "bear" : "bear",
    "zebra" : "zebra",
    "giraffe" : "giraffe",
    "backpack" : "backpack",
    "umbrella" : "umbrella",
    "handbag" : "handbag",
    "tie" : "tie",
    "suitcase" : "suitcase",
    "frisbee" : "frisbee",
    "skis" : "skis",
    "snowboard" : "snowboard",
    "sports ball" : "sports ball",
    "kite" : "kite",
    "baseball bat" : "baseball bat",
    "baseball glove" : "baseball glove",
    "skateboard" : "skateboard",
    "surfboard" : "surfboard",
    "tennis racket" : "tennis racket",
    "bottle" : "bottle",
    "wine glass" : "wineglass",
    "cup" : "cup",
    "fork" : "fork",
    "knife" : "knife",
    "spoon" : "spoon",
    "bowl" : "bowl",
    "banana" : "banana",
    "apple" : "apple",
    "sandwich" : "sandwich",
    "orange" : "orange",
    "broccoli" : "broccoli",
    "carrot" : "carrot",
    "hot dog" : "hotdog",
    "pizza" : "pizza",
    "donut" : "donut",
    "cake" : "cake",
    "chair" : "chair",
    "couch" : "couch",
    "potted plant" : "potted plant",
    "bed" : "bed",
    "dining table" : "dining table",
    "toilet" : "toilet",
    "tv" : "tv",
    "laptop" : "laptop",
    "mouse" : "mouse",
    "remote" : "remote",
    "keyboard" : "keyboard",
    "cell phone" : "cellphone",
    "microwave" : "microwave",
    "oven" : "oven",
    "toaster" : "toaster",
    "sink" : "sink",
    "refrigerator" : "refrigerator",
    "book" : "book",
    "clock" : "clock",
    "vase" : "vase",
    "scissors" : "scissors",
    "teddy bear" : "teddy",
    "hair drier" : "hairdryer",
    "toothbrush" : "toothbrush",
    }

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="coco",help="Dataset name.", choices=["voc", "coco"])
parser.add_argument("--model", type=str, default="glove", help="Text encoder model.", choices=["glove", "clip"])
parser.add_argument("--outdir", type=str, default="datasets", help="Output directory.")
args = parser.parse_args()


class_names = COCO_CATEGORIES if args.dataset == "coco" else VOC_CATEGORIES     
class_map = COCO_GLOVE_MAP if args.dataset == "coco" else VOC_GLOVE_MAP
    
if args.model == "clip":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(class_names).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        for idx, class_name in enumerate(class_names):
            print(text_features[idx].cpu().numpy())
            filePath = Path(f"{args.outdir}/{args.model}/{class_name}.txt")
            filePath.parent.mkdir(exist_ok=True, parents=True)
            filePath.touch(exist_ok=True)
            file = open(filePath,"w")
            np.savetxt(file, text_features[idx].cpu().numpy())
            file.close()
else:
    semantic_dim=300
    glove_vec = GloVe(name="6B", dim=semantic_dim)
    with torch.no_grad():
        for class_name in class_names:
            map_class = class_map[class_name]
            semantic_feature = torch.zeros(semantic_dim)
            for token in map_class:
                semantic_feature += glove_vec[token]
            semantic_feature /= len(map_class)
            print(class_name, semantic_feature.numpy())
            
            filePath = Path(f"{args.outdir}/{args.model}/{class_name}.txt")
            filePath.parent.mkdir(exist_ok=True)
            filePath.touch(exist_ok=True)
            file = open(filePath, "w")
            np.savetxt(file, semantic_feature.numpy())
            file.close()
            