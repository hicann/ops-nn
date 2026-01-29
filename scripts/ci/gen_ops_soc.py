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
import os
import sys
import re


def should_skip_directory(dir_name):
    """
    判断是否应该跳过该目录
    """
    skip_dirs = {
        'build', 'cmake', 'common', 'docs', 'examples',
        'experimental', 'scripts', 'tests', 'third_party'
    }
    return dir_name in skip_dirs


def parse_foreach_config(config_str):
    """
    解析 FOREACH_OPDEF 中的配置字符串
    """
    config_mapping = {
        'A2': 'ascend910b',
        '910_93': 'ascend910_93',
        'A5': 'ascend910_95',
        '910B': 'ascend910b',
        '910B_93': 'ascend910_93',
        '910B_95': 'ascend910_95',
        '910_95': 'ascend910_95',
        '910': 'ascend910',
        '910_55': 'ascend910_55',
    }

    found_configs = []
    config_str_upper = config_str.upper()

    priority_checks = [
        ('A2', 'ascend910b'),
        ('910_93', 'ascend910_93'),
        ('A5', 'ascend910_95'),
        ('910_55', 'ascend910_55'),
        ('910B', 'ascend910b'),
        ('910B_93', 'ascend910_93'),
        ('910B_95', 'ascend910_95'),
        ('910_95', 'ascend910_95'),
        ('910', 'ascend910'),
    ]

    for key, value in priority_checks:
        if key in config_str_upper and value not in found_configs:
            found_configs.append(value)

    return found_configs


def extract_static_map_configs(content):
    """
    从静态map中提取配置名称
    """
    configs = []

    map_patterns = [
        r'static\s+const\s+std::map<std::string[^>]*>\s+\w+\s*=\s*\{([^}]+)\}',
        r'\{"([a-zA-Z0-9_]+)"[^}]*\}',
    ]

    for pattern in map_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            config_matches = re.findall(r'"([a-zA-Z0-9_]+)"', match)
            for config in config_matches:
                if config not in configs:
                    configs.append(config)

    return list(set(configs))


def extract_set_ascend_config_calls(content):
    """
    提取 SetAscendConfig 调用中的配置名称
    """
    configs = []

    pattern1 = r'SetAscendConfig\([^,]+,\s*"([^"]+)"\)'
    pattern2 = r'SetAscendConfig\([^,]+,\s*"([^"]+)",\s*"([^"]+)"\)'

    matches1 = re.findall(pattern1, content)
    for match in matches1:
        if match not in configs:
            configs.append(match)

    matches2 = re.findall(pattern2, content)
    for match in matches2:
        version, dst_version = match
        if version not in configs:
            configs.append(version)
        if dst_version not in configs:
            configs.append(dst_version)

    return list(set(configs))


def extract_foreach_opdef_configs(content):
    """
    提取 FOREACH_OPDEF 相关格式的配置
    """
    configs = []

    pattern1 = r'FOREACH_OPDEF\(([^,]+),'
    matches1 = re.findall(pattern1, content)
    for match in matches1:
        config_str = match.strip()
        configs.extend(parse_foreach_config(config_str))

    pattern2 = r'FOREACH_OPDEF_END_([^(]+)\('
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        config_str = match.strip()
        configs.extend(parse_foreach_config(config_str))

    return list(set(configs))


def extract_traditional_aicore_configs(content):
    """
    提取传统的 AICore 配置名称
    """
    configs = []

    traditional_patterns = [
        r'this->AICore\(\)\.AddConfig\("([a-zA-Z0-9_]+)"',
        r'\.AddConfig\("([a-zA-Z0-9_]+)"',
        r'AddConfig\("([a-zA-Z0-9_]+)"',
    ]

    for pattern in traditional_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if match not in configs:
                configs.append(match)

    return configs


def extract_ai_core_configs(file_path):
    """
    从 _def.cpp 文件中提取 AICore 配置名称
    """
    configs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 方法1：匹配传统的 AICore 配置
        traditional_configs = extract_traditional_aicore_configs(content)
        if traditional_configs:
            configs.extend(traditional_configs)

        # 方法2：匹配 FOREACH_OPDEF 相关格式
        foreach_configs = extract_foreach_opdef_configs(content)
        if foreach_configs:
            configs.extend(foreach_configs)

        # 方法3：匹配静态map + SetAscendConfig 格式
        static_map_configs = extract_static_map_configs(content)
        set_ascend_configs = extract_set_ascend_config_calls(content)

        all_other_configs = list(set(static_map_configs + set_ascend_configs))
        if all_other_configs:
            configs.extend(all_other_configs)

        # 去重并返回
        return list(set(configs))

    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []


def ceil_div(a, b):
    return (a + b - 1) // b


def split_list_by_num_groups(lst, num_groups):
    avg = ceil_div(len(lst), num_groups)
    out = []
    last = 0.0
    for _ in range(num_groups):
        val = int(round(last + avg))
        out.append(lst[int(last):val])
        last = val
    return out

GROUPING_CONFIGS = {
    "default": {
        0: ["mat_mul_v3"],
        1: ["gemm_v3"],
        2: ["quant_batch_matmul_v3", "conv3d_v2"],
        3: ["weight_quant_batch_matmul_v2", "batch_mat_mul_v3", "apply_adam_w_v2"],
        4: [
            "scatter_elements_v2", "add_layer_norm", "layer_norm_grad_v3", 
            "masked_softmax_with_rel_pos_bias", "group_norm_grad",
            "group_norm_swish", "scatter_list", "group_norm_swish_grad"
        ],
    },
    "ascend910_95": {
        0: ["add_rms_norm_quant"],
        1: ["scatter_elements_v2"],
        2: ["quant_batch_matmul_v3"],
        3: ["extend_conv2d", "conv3d_v2"],
        4: ["conv2d_v2", "conv3d_transpose_v2"],
        5: ["quant_conv3d", "conv3d_backprop_input_v2", "apply_adam_w_v2"],
        6: ["batch_norm_grad_v3", "cross_entropy_loss_grad", "ascend_quant_v2", "dequant_swiglu_quant"],
        7: ["mat_mul_v3", "cross_entropy_loss", "weight_quant_batch_matmul_v2", "group_norm_grad"]
    }
}


def grouped(repository_path, soc, group_size):
    if soc in ("910_95", "ascend910_95"):
        config = GROUPING_CONFIGS.get("ascend910_95")
    else:
        config = GROUPING_CONFIGS.get("default")
    result = [[] for _ in range(len(config))]
    remain = []
    zero_tensor_num = 0

    for root, dirs, files in os.walk(repository_path):
        # 过滤掉不需要的目录
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]

        for file in files:
            if file.endswith('_def.cpp'):
                full_path = os.path.join(root, file)
                op_name = file.replace('_def.cpp', '')

                # 提取 AICore 配置
                ai_core_configs = extract_ai_core_configs(full_path)
                current_path = full_path
                for _ in range(3):
                    current_path = os.path.dirname(current_path)

                # 获取三层父目录的文件名
                parent_dir_name = os.path.basename(current_path)
                if soc in ai_core_configs:
                    matched = False
                    for idx, op_list in config.items():
                        if op_name in op_list:
                            result[idx].append(op_name)
                            matched = True
                            break
                    if not matched:
                        remain.append(op_name)
    
    filtered_result = []
    len_size = len(result)
    for i in range(len_size):
        if len(result[i]) == 0:
            zero_tensor_num += 1
        else:
            filtered_result.append(result[i])

    remain = sorted(remain)
    remain = split_list_by_num_groups(remain, group_size - len_size + zero_tensor_num if group_size > 8 else group_size)
    result.extend(remain)
    return result


def main(repository_path, soc, group_size):
    return grouped(repository_path, soc, group_size)
