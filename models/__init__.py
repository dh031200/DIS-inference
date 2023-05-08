import argparse
from pathlib import Path

import cv2
import gdown
import torch
import numpy as np
import torch.nn.functional as F
from yaml import safe_load
from torchvision.transforms.functional import normalize

from models.isnet import ISNetGTEncoder, ISNetDIS

model = 'isnet-general-use.pth'


def init_model():
    if not Path(model).exists():
        download_Model()
    net = ISNetDIS()
    net.load_state_dict(torch.load(model))
    net.to(device)
    net.eval()
    return net


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', default='assets/Puppies.jpg', help='source')
    _args = parser.parse_args()
    return _args


def get_name(args):
    return Path(args.source).stem


def save_dir(name):
    output_prefix = Path('output') / name
    output_prefix.mkdir(parents=True, exist_ok=True)
    return output_prefix


def read(args):
    return cv2.cvtColor(cv2.imread(args.source), cv2.COLOR_BGR2RGB)


def write(path, result):
    cv2.imwrite(path, result)


def pre_processing(image):
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
    im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), [1024, 1024], mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def post_processing(result, origin):
    result = torch.squeeze(F.interpolate(result[0][0], origin.shape[:2], mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    return (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)


def download_Model():
    gdown.download('https://drive.google.com/u/0/uc?id=1Js00mNhR8hQEsMJ3kbqE1CJthgPCqe41&export=download', model,
                   quiet=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = 'init_model', 'parse_args', 'get_name', 'read', 'write', 'pre_processing', 'post_processing', 'device', 'torch', 'save_dir'
