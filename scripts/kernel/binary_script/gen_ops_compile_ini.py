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

import sys
import re
from pathlib import Path


def parse_list(raw):
    entries = []
    for item in raw:
        item = item.strip()
        if not item:
            continue
        parts = item.split()
        if len(parts) < 2:
            continue
        op_name = parts[0]
        unit_name = parts[1]
        if len(parts) > 2 and "none" == parts[2].lower():
            parts[2] = "None"
        rest = " ".join(parts[2:])
        entries.append((op_name, unit_name, rest))
    return entries

config_dict = {}


def add_to_config(entries, key_name):
    for op_name_raw, unit_name_raw, value in entries:
        if unit_name_raw not in config_dict:
            config_dict[unit_name_raw] = {}
        if op_name_raw not in config_dict[unit_name_raw]:
            config_dict[unit_name_raw][op_name_raw] = {}
        config_dict[unit_name_raw][op_name_raw][key_name] = value


def main():
    if len(sys.argv) < 5:
        raise SystemExit(0)

    output_dir = Path(sys.argv[1])
    simplified_key_str = sys.argv[2]
    impl_mode_str = sys.argv[3]
    auto_sync_str = sys.argv[4]
    options_str = sys.argv[5]

    simplified_key_list = simplified_key_str.split("/") if simplified_key_str else []
    impl_mode_list = impl_mode_str.split("/") if impl_mode_str else []
    auto_sync_list = auto_sync_str.split("/") if auto_sync_str else []
    options_list = options_str.split("/") if options_str else []
    output_dir.mkdir(exist_ok=True)

    simplified_keys = parse_list(simplified_key_list)
    impl_modes = parse_list(impl_mode_list)
    auto_syncs = parse_list(auto_sync_list)
    op_options = parse_list(options_list)

    add_to_config(simplified_keys, "simplified_key")
    add_to_config(impl_modes, "impl_mode")
    add_to_config(auto_syncs, "auto_sync")
    add_to_config(op_options, "options")

    # 每个单元一个文件，文件内容是算子的配置项
    for unit_name, op_configs in config_dict.items():
        ini_path = output_dir / f"kernel-options-{unit_name}.ini"
        with open(ini_path, "w", encoding="utf-8") as f:
            for op_name, config in op_configs.items():
                f.write(f"[{op_name}]\n")
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
                f.write("\n")


if __name__ == "__main__":
    main()