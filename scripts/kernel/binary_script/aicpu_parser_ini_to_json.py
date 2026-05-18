#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import json
import logging as log
import os
import stat
import sys


VALID_PARAM_TYPES = {"dynamic", "optional", "required"}
VALID_SUB_TYPES = {"1", "2", "3", "4"}
VALID_OPS_FLAGS = {"OPS_FLAG_OPEN", "OPS_FLAG_CLOSE"}
INPUT_PREFIXES = ("input", "dynamic_input", "optional_input")
OUTPUT_PREFIXES = ("output", "dynamic_output", "optional_output")


def parse_ini_files(ini_files):
    """
    parse ini files to json
    Parameters:
    ----------------
    ini_files:input file list
    return:ops_info
    ----------------
    """
    tbe_ops_info = {}
    for ini_file in ini_files:
        if not os.path.exists(ini_file):
            log.warning("ini file %s not exists!", ini_file)
            continue
        parse_ini_to_obj(ini_file, tbe_ops_info)
    return tbe_ops_info


def parse_ini_to_obj(ini_file, tbe_ops_info):
    """
    parse ini file to json obj
    Parameters:
    ----------------
    ini_file:ini file path
    tbe_ops_info:ops_info
    ----------------
    """
    current_op = {}
    current_op_name = ""
    with open(ini_file) as ini_file_handle:
        for raw_line in ini_file_handle:
            line = raw_line.rstrip()
            if not line:
                continue
            if line.startswith("["):
                current_op_name = line[1:-1]
                current_op = {}
                tbe_ops_info[current_op_name] = current_op
                continue
            save_op_entry(current_op_name, current_op, line)


def save_op_entry(op_name, op_info, line):
    key_text, value_text = split_ini_line(line)
    section_key, field_key = key_text.split(".", 1)
    section = op_info.setdefault(section_key, {})
    if field_key in section:
        raise RuntimeError("Op:" + op_name + " " + section_key + " " + field_key + " is repeated!")
    section[field_key] = value_text


def split_ini_line(line):
    key_text, _, value_text = line.partition("=")
    return key_text.strip(), value_text.strip()


def is_aicpu_op(op):
    op_info = op.get("opInfo", {})
    return op_info.get("engine") == "DNN_VM_AICPU"


def is_input_key(op_info_key):
    return any(op_info_key.startswith(prefix) for prefix in INPUT_PREFIXES)


def is_output_key(op_info_key):
    return any(op_info_key.startswith(prefix) for prefix in OUTPUT_PREFIXES)


def check_param_type(op_key, op_info_key, op_io_info):
    if "paramType" not in op_io_info:
        return True
    if op_io_info["paramType"] not in VALID_PARAM_TYPES:
        log.error("op: %s %s paramType not valid, valid key:[dynamic, optional, required]",
                  op_key, op_info_key)
        return False
    return True


def check_missing_keys(op_key, op_info_key, op_io_info, required_keys):
    missing_keys = []
    for required_key in required_keys:
        if required_key not in op_io_info:
            missing_keys.append(required_key)
    if len(missing_keys) == 0:
        return True
    log.error("op: %s %s missing: %s", op_key, op_info_key, ",".join(missing_keys))
    return False


def check_io_info(op_key, op_info_key, op_io_info, required_keys):
    is_valid = check_missing_keys(op_key, op_info_key, op_io_info, required_keys)
    if not check_param_type(op_key, op_info_key, op_io_info):
        is_valid = False
    return is_valid


def check_aicpu_extend_cfg(op_key, op):
    op_info = op.get("opInfo", {})
    valid = True

    subtype = op_info.get("subTypeOfInferShape")
    if subtype is not None and subtype not in VALID_SUB_TYPES:
        log.error("op: %s opInfo.subTypeOfInferShape not valid, valid key:[1, 2, 3, 4]", op_key)
        valid = False

    ops_flag = op_info.get("opsFlag")
    if ops_flag is not None and ops_flag not in VALID_OPS_FLAGS:
        log.error("op: %s opInfo.opsFlag not valid, valid key:[OPS_FLAG_OPEN, OPS_FLAG_CLOSE]", op_key)
        valid = False

    workspace_size = op_info.get("workspaceSize")
    if workspace_size is not None:
        if not workspace_size.isdigit():
            log.error("op: %s opInfo.workspaceSize not valid, should be integer in [100, 500]", op_key)
            valid = False
        else:
            value = int(workspace_size)
            if value < 100 or value > 500:
                log.error("op: %s opInfo.workspaceSize out of range, expected [100, 500]", op_key)
                valid = False

    kernel_so = op_info.get("kernelSo")
    if kernel_so is not None and not kernel_so.endswith(".so"):
        log.error("op: %s opInfo.kernelSo not valid, should end with .so", op_key)
        valid = False
    return valid


def check_aicpu_io_info(op_key, op):
    required_input_keys = ["name"]
    required_output_keys = ["name"]
    is_valid = True
    for op_info_key, op_io_info in op.items():
        if is_input_key(op_info_key):
            if not check_io_info(op_key, op_info_key, op_io_info, required_input_keys):
                is_valid = False
        elif is_output_key(op_info_key):
            if not check_io_info(op_key, op_info_key, op_io_info, required_output_keys):
                is_valid = False
    return is_valid


def check_aicpu_op(op_key, op):
    if not is_aicpu_op(op):
        log.error("op: %s opInfo.engine not valid, expected DNN_VM_AICPU", op_key)
        return False

    is_valid = check_aicpu_extend_cfg(op_key, op)
    if not check_aicpu_io_info(op_key, op):
        is_valid = False
    return is_valid


def check_op_info(tbe_ops):
    """
    Check info info
    """
    log.info("==============check valid for ops info start==============")
    is_valid = True
    for op_key in tbe_ops:
        if not check_aicpu_op(op_key, tbe_ops[op_key]):
            is_valid = False

    log.info("==============check valid for ops info end================")
    return is_valid


def write_json_file(tbe_ops_info, json_file_path):
    """
    Save info to json file
    Parameters:
    ----------------
    tbe_ops_info: ops_info
    json_file_path: json file path
    ----------------
    """
    json_file_real_path = os.path.realpath(json_file_path)
    with open(json_file_real_path, "w") as file_handle:
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP + stat.S_IRUSR)
        json.dump(tbe_ops_info, file_handle, sort_keys=True, indent=4, separators=(',', ':'))
    log.info("Compile op info cfg successfully.")


def parse_ini_to_json(ini_file_paths, outfile_path):
    """
    parse ini files to json file
    Parameters:
    ----------------
    ini_file_paths: list of ini file path
    outfile_path: output file path
    ----------------
    """
    tbe_ops_info = parse_ini_files(ini_file_paths)
    if not check_op_info(tbe_ops_info):
        log.error("Compile op info cfg failed.")
        return False
    write_json_file(tbe_ops_info, outfile_path)
    return True


def parse_cli_args(args):
    output_path = "tbe_ops_info.json"
    ini_files = []
    for arg in args[1:]:
        suffix = os.path.splitext(arg)[1]
        if suffix == ".ini":
            ini_files.append(arg)
            output_path = os.path.splitext(arg)[0] + ".json"
            continue
        if suffix == ".json":
            output_path = arg
    if not ini_files:
        ini_files = ["tbe_ops_info.ini"]
    return ini_files, output_path


def main():
    log.basicConfig(
        stream=sys.stdout,
        format=f'{os.path.basename(__file__)}: %(levelname)s: %(message)s',
        level=log.INFO
    )
    ini_file_path_list, output_file_path = parse_cli_args(sys.argv)
    if parse_ini_to_json(ini_file_path_list, output_file_path):
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
