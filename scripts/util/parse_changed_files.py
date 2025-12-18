#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
import logging
from dependency_parser import OpDependenciesParser


OP_CATEGORY_LIST = []
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class UtMatcher():
    def __init__(self):
        self.ops = set()


    def match(self, changed_file):
        changed_file = str(os.path.relpath(changed_file, os.getenv('BASE_PATH'))).strip()
        files = changed_file.split('/')
        if len(files) > 2 and files[0] in OP_CATEGORY_LIST:
            op_name = files[1]
            if op_name == 'common':
                op_name = files[0] + '.common'
        else:
            return False
        if self._ut_match(changed_file):
            self.ops.add(op_name)
            return True
        return False


    def _ut_match(self, changed_file):
        pass


class OpApiUt(UtMatcher):
    def _ut_match(self, changed_file):
        return (
            changed_file.find('op_api') != -1 and changed_file.find('.txt') == -1
            or (changed_file.find('op_host') != -1 and changed_file.find('test_aclnn_') != -1)
        )


class OpHostUt(UtMatcher):
    def _ut_match(self, changed_file):
        return changed_file.find('op_host') != -1 and changed_file.find('config') == -1 and \
            changed_file.find('test_aclnn_') == -1 and changed_file.find('.txt') == -1


class OpKernelUt(UtMatcher):
    def _ut_match(self, changed_file):
        return changed_file.find('op_kernel') != -1


class OpGraphUt(UtMatcher):
    def _ut_match(self, changed_file):
        return changed_file.find('op_graph') != -1


UT_MATCHERS = {'op_api_ut': OpApiUt(), 'op_host_ut': OpHostUt()}


if __name__ == '__main__':
    changed_files_info_file = sys.argv[1]
    changed_files = []
    parser = OpDependenciesParser(os.getenv('BUILD_PATH'))
    OP_CATEGORY_LIST.extend(parser.get_category_list())
    if not os.path.exists(changed_files_info_file):
        logging.error("[ERROR] change file is not exist, can not get file change info in this pull request.")
        exit(1)
    with open(changed_files_info_file) as or_f:
        changed_files = or_f.readlines()
    for changed_file in changed_files:
        if not os.path.exists(r'{}'.format(changed_file.strip())):
            continue
        for ut_matcher in UT_MATCHERS.values():
            if ut_matcher.match(changed_file):
                break
    for key, matcher in UT_MATCHERS.items():
        if matcher.ops:
            (_, reverse_op_dependencies) = parser.get_dependencies_by_ops(matcher.ops) # 获取反向依赖
            (op_dependencies, _) = parser.get_dependencies_by_ops(reverse_op_dependencies) # 根据反向依赖，获取编译依赖
            compile_ops = ';'.join(list(set(op_dependencies)))
            print('%s:%s:%s ' % (key, ';'.join(reverse_op_dependencies), compile_ops))