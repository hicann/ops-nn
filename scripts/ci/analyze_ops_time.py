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

import os
import re
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.join(script_dir, "../../build/")
build_dir = os.path.abspath(build_dir)
workdir = os.path.join(build_dir, "binary")
workdir = os.path.abspath(workdir)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

MAIN_FUNC_RE = re.compile(r'--main_func=([^&\s]+)')
EXE_TIME_RE = re.compile(r'exe_time:\s*(\d+)\s*s')

# 遍历build_logs
log_dirs = []
for child in Path(workdir).iterdir():
    if not child.is_dir():
        continue
    
    bin_dir = child / "bin"
    build_logs_dir = bin_dir / "build_logs"
    if build_logs_dir.exists():
        log_dirs.append(build_logs_dir)
        
all_stats = defaultdict(lambda: defaultdict(lambda: {
    'earliest_start': None,
    'latest_end': None,
    'all': 0,                   # opc总数
    'fail': 0,                  # 失败的opc个数
    'obj_size_bytes': 0,        # .o文件总大小
    'obj_dir_checked': False
}))

for log_dir_path in log_dirs:
    soc_name = log_dir_path.parent.parent.name
    if not soc_name:
        print(f"Unrecognized soc: {log_dir_path}")
        continue
    
    # .o所在目录为build/binary/{soc}/bin/{soc}/
    ops_root = log_dir_path.parent / soc_name
    if not ops_root.exists():
        print(f"Ops dir not exist: {ops_root}")
    
    for log_file in log_dir_path.iterdir():
        if not log_file.is_file():
            continue
        if not log_file.suffix.lower() == '.log' or log_file.name.startswith('.'):
            continue
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"read file fail: {log_file}, {e}")
            continue
        if len(lines) < 1:
            continue
        
        # 约定：第一行一定以时间戳+Build started开始，最后一行以时间戳+exe_time结束 
        # 参照scripts\kernel\binary_script\build_binary_op_exe_task.sh
        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        
        start_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+Build started:', first_line)
        main_func_match = MAIN_FUNC_RE.search(first_line)
        if not start_match or not main_func_match:
            continue
        
        start_time = datetime.strptime(start_match.group(1), TIME_FORMAT)
        op = main_func_match.group(1)
        if op == "conv3dv2":
            op = "conv3d_v2"
        elif op == "conv2dv2":
            op = "conv2d_v2"
        elif op == "dynamic_rnn_v2":
            op = "dynamic_rnnv2"
        all_stats[soc_name][op]['all'] += 1
        
        has_valid_end = False
        if len(lines) >= 2:
            last_line = lines[-1].strip()
            end_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', last_line)
            exe_match = EXE_TIME_RE.search(last_line)
            if end_match and exe_match:
                end_time = datetime.strptime(end_match.group(1), TIME_FORMAT)
                stats = all_stats[soc_name][op]
                if stats['earliest_start'] is None or start_time < stats['earliest_start']:
                    stats['earliest_start'] = start_time
                if stats['latest_end'] is None or end_time > stats['latest_end']:
                    stats['latest_end'] = end_time
                has_valid_end = True
        
        if not has_valid_end:
            all_stats[soc_name][op]['fail'] += 1
        
        stats = all_stats[soc_name][op]
        if not stats['obj_dir_checked']:
            stats['obj_dir_checked'] = True
            obj_dir = ops_root / op
            total_size = 0
            if obj_dir.exists():
                for o_file in obj_dir.rglob("*.o"):
                    if o_file.is_file():
                        try:
                            total_size += o_file.stat().st_size
                        except (OSError, ValueError):
                            pass
            stats['obj_size_bytes'] = total_size

for soc_name, op_dict in all_stats.items():
    csv_rows = []
    for op, data in op_dict.items():
        all_count = data['all']
        fail_count = data['fail']
        success_count = all_count - fail_count
        obj_size = data['obj_size_bytes']
        
        if success_count > 0:
            duration_s = int((data['latest_end'] - data['earliest_start']).total_seconds())
        else:
            duration_s = -1
        
        csv_rows.append({
            'op': op,
            'duration(s)': duration_s,
            'size': obj_size,
            'all': all_count
        })
    
    # 按照有效时长排序（有时长的在前），然后按照时长降序排序。
    csv_rows.sort(key=lambda x: (x['duration(s)'] != -1, x['duration(s)']), reverse=True)
    
    csv_path = Path(build_dir) / f"ops_cost_{soc_name}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=[
                'op', 'duration(s)', 'size', 'all'
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"The {soc_name} statistical results have been saved to {csv_path}")