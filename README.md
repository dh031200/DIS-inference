# DIS-inference

[![PyPI - Version](https://img.shields.io/pypi/v/dis-inference.svg)](https://pypi.org/project/dis-inference)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dis-inference.svg)](https://pypi.org/project/dis-inference)

Inference implementation of Dichotomous Image Segmentation

## [Highly Accurate Dichotomous Image Segmentation (ECCV 2022)](https://arxiv.org/pdf/2203.03041.pdf)

#### [Xuebin Qin](https://xuebinqin.github.io/), [Hang Dai](https://scholar.google.co.uk/citations?user=6yvjpQQAAAAJ&hl=en), [Xiaobin Hu](https://scholar.google.de/citations?user=3lMuodUAAAAJ&hl=en), [Deng-Ping Fan*](https://dengpingfan.github.io/), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en).

[**Project Page**](https://xuebinqin.github.io/dis/index.html), [**Arxiv**](https://arxiv.org/pdf/2203.03041.pdf), [**中文
**](https://github.com/xuebinqin/xuebinqin.github.io/blob/main/ECCV2022_DIS_Chinese.pdf).

<br>

| Origin                                                                                   | DIS                                                                                         |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| ![Before](https://github.com/dh031200/DIS-inference/blob/main/assets/Lenna.png?raw=true) | ![After](https://github.com/dh031200/DIS-inference/blob/main/assets/Lenna_dis.png?raw=true) |

<br>

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Reference](#reference)

## Installation

```console
pip install dis-inference
```

## Usage

### CLI

#### **command**:  `dis-inference`
#### **arguments**:

* **--silent(optional)** : Whether to print verbose.
* **Source image(mandatory)** : Source image path.

```console
> dis-inference Lenna.png
Output saved as `Lenna_dis.png`
> dis-inference --silent Lenna.png
```

### Python

```python
from dis_inference import inference

# 1. Inference with path
output = inference('Lenna.png')
cv2.imwrite('Lenna_dis.png', output)

# 2. Inference cv2 image
image = cv2.imread('Lenna.png')
output = inference(image)
cv2.imwrite('Lenna_dis.png', output)

# 3. Inference cv2 image with save parameter
image = cv2.imread('Lenna.png')
output = inference(image, save=True, output='Lenna_dis.png')

# 4. With save parameter
image = inference('Lenna.png', save=True)
```

## License

`dis-inference` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)
license.

## Reference

https://github.com/xuebinqin/DIS

## Citation

```
@InProceedings{qin2022,
      author={Xuebin Qin and Hang Dai and Xiaobin Hu and Deng-Ping Fan and Ling Shao and Luc Van Gool},
      title={Highly Accurate Dichotomous Image Segmentation},
      booktitle={ECCV},
      year={2022}
}
```
