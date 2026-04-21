#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""合并算子binary_info_config.json。"""

import argparse
import json
import os
import sys
from typing import List


def load_json_file(json_file: str):
    """加载json文件。"""
    with open(json_file, encoding='utf-8') as file:
        json_content = json.load(file)
    return json_content


def save_json_file(output_file: str, content):
    """保存json文件。"""
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=True, indent=2)


def update_config(base_content, update_content):
    """更新配置。"""
    new_content = base_content.copy()
    new_content.update(update_content)
    return dict(sorted(new_content.items()))


def merge_files(files: List[str], priority: str = "last"):
    """批量合并多个文件。"""
    if not files:
        return {}
    if len(files) == 1:
        return load_json_file(files[0])
    result = load_json_file(files[0])
    for f in files[1:]:
        other = load_json_file(f)
        if priority == "first":
            result = update_config(other, result)
        else:
            result = update_config(result, other)
    return result


def parse_args(argv: List[str]):
    """入参解析。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-file', help='the basic binary_info_config file')
    parser.add_argument('--update-file', help='the update binary_info_config file')
    parser.add_argument('--input-files', nargs='+', help='batch merge multiple files')
    parser.add_argument('--output-file', required=True, type=os.path.realpath,
                        help='the output binary_info_config file')
    parser.add_argument('--priority', choices=['first', 'last'], default='last',
                        help='merge priority')
    args = parser.parse_args(argv)
    return args


def main(argv: List[str]) -> bool:
    """主流程。"""
    args = parse_args(argv)
    if args.input_files:
        result = merge_files(args.input_files, args.priority)
    else:
        # 加载基础文件和待合并文件
        base_content = load_json_file(args.base_file)
        update_content = load_json_file(args.update_file)
        # 合并binList
        result = update_config(base_content, update_content)
    # 保存结果           
    save_json_file(args.output_file, result)
    return True


if __name__ == '__main__':
    if not main(sys.argv[1:]):  # pragma: no cover
        sys.exit(1)  # pragma: no cover