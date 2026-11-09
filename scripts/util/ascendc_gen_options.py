#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import stat
import os


def write_options_to_file(
    file_name: str, options_str: str, op_type: str, compute_unit: str, split_char: str
):
    descriptor = os.open(
        file_name, os.O_WRONLY | os.O_CREAT | os.O_APPEND, stat.S_IWUSR | stat.S_IRUSR
    )
    with os.fdopen(descriptor, "a") as stream:
        stream.write(split_char.join((op_type, compute_unit, options_str)) + "\n")


def split_compile_options(options):
    ordinary, debug, tiling = [], [], ""
    debug_flags = {"--oom": "oom", "--save-temp-files": "dump_cce"}
    for option in options:
        if option in debug_flags:
            debug.append(debug_flags[option])
        elif option.startswith("--tiling_key="):
            tiling = option.partition("=")[2].replace(",", ";")
        else:
            ordinary.append(option)
    configuration = []
    if debug:
        configuration.append("--op_debug_config=" + ";".join(debug))
    if tiling:
        configuration.append("--tiling_key=" + tiling)
    return ordinary, "@".join(configuration)


def gen_compile_options(
    compile_options_file: str, op_type: str, compute_unit: str, compile_options: list
):
    ordinary, configuration = split_compile_options(compile_options)
    if ordinary:
        write_options_to_file(
            compile_options_file, ";".join(ordinary), op_type, compute_unit, ","
        )
    if configuration:
        options_file = os.path.join(
            os.path.dirname(compile_options_file), "custom_opc_options.ini"
        )
        write_options_to_file(options_file, configuration, op_type, compute_unit, "@")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise RuntimeError("arguments must greater than 4")
    compute_units, options = [], []
    for argument in sys.argv[3:]:
        if argument.upper().startswith("ASCEND"):
            compute_units.append(argument)
        else:
            options.append(argument)
    gen_compile_options(sys.argv[1], sys.argv[2], ";".join(compute_units), options)
