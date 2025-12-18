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


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
OP_CATEGORY_LIST = []
BASE_PATH = os.getenv('BASE_PATH')



if __name__ == '__main__':
    changed_files_info_file = sys.argv[1]
    changed_files = []
    if not os.path.exists(changed_files_info_file):
        logging.error("[ERROR] change file is not exist, can not get file change info in this pull request.")
        exit(1)
    with open(changed_files_info_file) as or_f:
        changed_files = or_f.readlines()
    parser = OpDependenciesParser(os.getenv('BUILD_PATH'))
    OP_CATEGORY_LIST.extend(parser.get_category_list())
    ops = set()
    for changed_file in changed_files:
        if not os.path.exists(r'{}'.format(changed_file.strip())):
            continue
        changed_file = str(os.path.relpath(changed_file, BASE_PATH)).strip()
        files = changed_file.split('/')
        if len(files) > 2 and files[0] in OP_CATEGORY_LIST:
            op_name = files[1]
            if op_name == 'common':
                op_name = files[0] + '.common'
            ops.add(op_name)

    (_, reverse_op_dependencies) = parser.get_dependencies_by_ops(ops) # 获取反向依赖
    (op_dependencies, _) = parser.get_dependencies_by_ops(reverse_op_dependencies) # 根据反向依赖，获取编译依赖
    compile_ops = ';'.join(list(set(op_dependencies)))
    print('%s:%s' % (';'.join(reverse_op_dependencies), compile_ops))
