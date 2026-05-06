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

import os
import sys


def insert_ini_kernel_src(kernel_srcs, src_path, compute_unit):
    items = kernel_srcs.strip().split(',')
    if not items:
        sys.exit()
    
    ini_filename = f"aic-{compute_unit}-ops-info.ini"
    ini_path = os.path.join(src_path, ini_filename)
    
    if not os.path.exists(ini_path):
        sys.exit()
        
    with open(ini_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for item in items:
        item = item.strip()
        if not item:
            continue
    
        parts = item.split()
        if len(parts) < 3:
            continue
        
        op_name = parts[0]
        op_compute_unit = parts[1]
        arch_path = parts[2]
        
        default_kernel_src = False
        if op_compute_unit != compute_unit:
            if op_compute_unit == "ALL":
                default_kernel_src = True
            else:
                continue
        
        section_name = f"[{op_name}]"
        found_section = False
        section_end_index = -1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line == section_name:
                found_section = True
                continue
            
            if not found_section:
                continue
            
            if line.startswith('[') and line.endswith(']'):
                section_end_index = i
                break
            else:
                section_end_index = len(lines)
        
        if not found_section:
            continue
        
        kernel_src_found = False
        for i in range(lines.index(f"{section_name}\n"), section_end_index):
            line = lines[i].strip()
            if line.startswith("kernelSrc.value=") and not default_kernel_src:
                old_value = line.split('=', 1)[1].strip()
                lines[i] = f"kernelSrc.value={arch_path}\n"
                kernel_src_found = True
                break
            
        if not kernel_src_found:
            insert_pos = lines.index(f"{section_name}\n") + 1
            new_line = f"kernelSrc.value={arch_path}\n"
            lines.insert(insert_pos, new_line)
    with open(ini_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit()
    
    kernel_srcs = sys.argv[1]
    src_path = sys.argv[2]
    compute_unit = sys.argv[3]
    
    insert_ini_kernel_src(kernel_srcs, src_path, compute_unit)