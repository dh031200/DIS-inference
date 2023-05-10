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
@click.option('--silent', '-s', is_flag=True, help='Print verbose')
@click.argument('source')
def dis_inference(source, silent):
    main(source, silent)


def infer(src):
    net = init_model()
    image = pre_processing(src).to(device)
    result = net(image)
    output = post_processing(result, src)
    return output


def main(source, silent):
    src_name, extension = get_name(source)
    src = read(source)
    result = infer(src)
    write(f"{src_name}_dis{extension}", result, silent)
