import json
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os.path as op
import random


train_img_dir = 'path-to-the-imageDir-of-your-own-Dataset' ## for example -> '/kaggle/input/coco-2017-dataset/coco2017/train2017'
train_annotation_file = 'path-to-the-captionDir-of-your-own-Dataset' ## for example -> '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
input_resolution = 224

def read_json(fname):
    """Read a JSON file and return its contents as a dictionary."""
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])

def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)
    return img_id_to_captions

class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""

    def __init__(self, text_tokenizer, context_length=77, input_resolution=224):
        super(CLIP_COCO_dataset, self).__init__()
        annotation_file = train_annotation_file
        annotations = read_json(annotation_file)
        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        self.img_id_to_captions = get_img_id_to_captions(annotations)
        self.img_ids = list(self.img_id_to_filename.keys())
        self.img_dir = train_img_dir
        self.transform = _transform(input_resolution)
        self.context_length = context_length
        self._tokenizer = text_tokenizer

    def tokenize(self, text):
        tokens = self._tokenizer(
            text, 
            max_length=self.context_length, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokens["input_ids"].squeeze()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        text = random.choice(self.img_id_to_captions[img_id])
        img_filename = self.img_id_to_filename[img_id]
        img_path = op.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenize(text)
        return img_input, text_input
