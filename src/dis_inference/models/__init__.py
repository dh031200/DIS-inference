# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
import os
import platform
import tempfile
from typing import Union
from pathlib import Path

import cv2
import gdown
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from .isnet import ISNetDIS


def init_model():
    if not Path(model).exists():
        download_Model()
    net = ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model))
    else:
        net.load_state_dict(torch.load(model, map_location=device))
    net.to(device)
    net.eval()
    return net


def get_name(source):
    return Path(source).stem + '_dis', Path(source).suffix


def read(source):
    return cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)


def write(path, result, silent):
    if not silent:
        print(f'Output saved as `{path}`')
    cv2.imwrite(path, result)


def check_params(source, output):
    if type(source) == str:
        if output:
            extension = (
                '.png' if not any([output.endswith('png'), output.endswith('jpg'), output.endswith('jpeg')]) else ''
            )
        else:
            output, extension = get_name(source)
        source = read(source)
    else:
        if not output:
            output = 'output'
        extension = (
            '.png' if not any([output.endswith('.png'), output.endswith('.jpg'), output.endswith('.jpeg')]) else ''
        )
    return source, output, extension


def pre_processing(image):
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
    im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), [1024, 1024], mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def post_processing(result, origin):
    result = torch.squeeze(F.interpolate(result[0][0], origin.shape[:2], mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    return (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)


def download_Model():
    gdown.download(
        "https://drive.google.com/u/0/uc?id=1Js00mNhR8hQEsMJ3kbqE1CJthgPCqe41&export=download",
        model,
        quiet=False,
    )


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    try:
        with tempfile.TemporaryFile(dir=dir_path):
            pass
        return True
    except OSError:
        return False


def get_user_config_dir(sub_dir='DIS-inference'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        Path: The path to the user config directory.
    """
    # Return the appropriate config directory for each operating system
    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(str(path.parent)):
        path = Path('/tmp') / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans
USER_CONFIG_DIR = Path(os.getenv('DIS_CONFIG_DIR', get_user_config_dir()))
model = str(USER_CONFIG_DIR / "isnet-general-use.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = (
    "init_model",
    "check_params",
    "write",
    "pre_processing",
    "post_processing",
    "device",
    "torch",
    "USER_CONFIG_DIR",
)
