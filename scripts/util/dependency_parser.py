#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import argparse
import logging
import subprocess


KEYS = ['OP_CATEGORY', 'OP_NAME', 'HOSTNAME', 'MODE', 'DIR', 'OPTYPE', 'ACLNNTYPE', 'DEPENDENCIES']
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ops',
                        nargs='?',
                        required=True,
                        help='Operators that need to find dependency.')

    parser.add_argument('-p',
                        '--path',
                        nargs='?',
                        required=True,
                        help='Build path.')

    return parser.parse_args()


def set_dict_value(dict_value, key, value):
    if key not in dict_value:
        dict_value[key] = []
    dict_value[key].append(value)


class OpDependenciesParser:
    def __init__(self, build_path):
        self.all_ops_dependency = {}
        self.all_ops_reverse_dependency = {}
        self.all_ops = []
        self.all_category_ops = {}
        self.parse_dependency(build_path)
        pass

    def find_all_dependency(self, op, result_dependencies, all_dependencies, src_op):
        if op not in self.all_ops:
            logging.error('%s is not exists, please check.', op)
            raise RuntimeError(f"{op} is not exists, please check.")
        if op in result_dependencies:
            return
        result_dependencies.append(op)
        for sub_op in all_dependencies.get(op, []):
            self.find_all_dependency(sub_op, result_dependencies, all_dependencies, src_op)


    def parse_line(self, line):
        last_key = None
        op_type = None
        op_category = None
        common_name = None
        for value in line.strip().split(';'):
            if value in KEYS:
                last_key = value
                continue
            if last_key == 'OP_CATEGORY':
                op_category = value
                common_name = op_category + '.common'
            if last_key == 'OP_NAME':
                op_type = value
                if op_type == 'common':
                    op_type = common_name
                set_dict_value(self.all_category_ops, op_category, op_type)
                self.all_ops.append(op_type)
            if last_key == 'DEPENDENCIES':
                set_dict_value(self.all_ops_dependency, op_type, value)
                set_dict_value(self.all_ops_reverse_dependency, value, op_type)


    def parse_dependency(self, build_path):
        build_path = os.path.abspath(build_path)
        file_path = os.path.join(build_path, 'tmp', 'ops_config.txt')
        if not os.path.exists(file_path):
            logging.error('%s config file is not exists.', file_path)
            raise RuntimeError(f"{file_path} config file is not exists.")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.parse_line(line)
        for (op_category, ops) in self.all_category_ops.items():
            common_name = op_category + '.common'
            self.all_ops.append(common_name)
            if common_name in ops:
                self.all_ops_reverse_dependency[common_name] = []
                self.all_ops_reverse_dependency[common_name].extend(ops)
                for op in ops:
                    set_dict_value(self.all_ops_dependency, op, common_name)


    def get_dependencies_by_ops(self, ops):
        result_ops = []
        reverse_ops = []
        for op in ops:
            if op not in reverse_ops:
                self.find_all_dependency(op, reverse_ops, self.all_ops_reverse_dependency, op)
            if op not in result_ops:
                self.find_all_dependency(op, result_ops, self.all_ops_dependency, op)
        return (result_ops, reverse_ops)


    def get_category_list(self):
        return self.all_category_ops.keys()


def main():
    args = args_parse()
    parser = OpDependenciesParser(args.path)
    (op_dependencies, reverse_op_dependencies) = parser.get_dependencies_by_ops(args.ops.split(';'))
    op_dependencies = ';'.join(op_dependencies)
    reverse_op_dependencies = ';'.join(reverse_op_dependencies)
    logging.info('op_dependencies:%s, reverse_op_dependencies:%s', op_dependencies, reverse_op_dependencies)
    subprocess.run(['cmake', '-DASCEND_COMPILE_OPS=' + op_dependencies, '..'], cwd=args.path)


if __name__ == '__main__':
    main()
