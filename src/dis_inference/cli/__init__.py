# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
from typing import Union

import click
import numpy as np

from dis_inference.__about__ import __version__
from ..models import (
    init_model,
    pre_processing,
    device,
    post_processing,
    get_name,
    read,
    write,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="DIS-inference")
@click.option('--silent', '-s', is_flag=True, help='whether print verbose')
@click.argument('source')
def dis_inference(source, silent=False):
    inference(source, save=True, silent=silent)


def inference(source: Union[str, np.ndarray], save=False, silent=False, output='output'):
    """
    :param source: Source image for inference.
    :param save: Whether to save output image.
    :param silent: Whether to print verbose.
    :param output: The name of output image file
    :return: (numpy.ndarray)dichotomous segmentation image
    """
    if type(source) == str:
        src_name, extension = get_name(source)
        source = read(source)
    else:
        src_name, extension = output, '.png'
    net = init_model()
    image = pre_processing(source).to(device)
    result = net(image)
    output = post_processing(result, source)
    if save:
        write(f"{src_name}_dis{extension}", output, silent)
    return output
