import torch
from pathlib import Path
from collections import OrderedDict
from transformers import CLIPTokenizer
from dataset import CLIP_COCO_dataset, get_dataloader
from torch.utils.data import Dataset, DataLoader
from train.trainer import CLIPTrainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize 
clip_coco_dataset = CLIP_COCO_dataset(CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32'))
train_loader = DataLoader(clip_coco_dataset, batch_size=32, shuffle=True, num_workers=4)

ct = CLIPTrainer(num_epochs=100)
loss_history = ct.train_model(train_loader)