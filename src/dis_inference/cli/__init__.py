# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
import click

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


def inference(source: str, save=False, silent=False):
    """
    :param source: Source image for inference.
    :param save: Whether to save output image.
    :param silent: Whether to print verbose.
    :return: (numpy.ndarray)dichotomous segmentation image
    """
    net = init_model()
    src = read(source)
    image = pre_processing(src).to(device)
    result = net(image)
    output = post_processing(result, src)
    if save:
        src_name, extension = get_name(source)
        write(f"{src_name}_dis{extension}", output, silent)
    return output
