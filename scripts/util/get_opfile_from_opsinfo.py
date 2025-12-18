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

"""
get_opfile_from_opsinfo.py
"""
import sys
import os
import json


if __name__ == '__main__':
    ops_info_file = sys.argv[1]
    op_type = sys.argv[2]

    if os.path.exists(ops_info_file):
        with open(ops_info_file, "r") as file_op:
            ops_info_json = json.load(file_op)
            op_info = ops_info_json.get(op_type)
            if op_info is not None:
                op_file = op_info.get("opFile")
                if op_file is not None:
                    op_file_name = op_file.get("value")
                    if op_file_name is not None:
                        print(op_file_name)
