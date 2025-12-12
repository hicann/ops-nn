#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import sys
import numpy as np
import glob
import os
import filecmp

curr_dir = os.path.dirname(os.path.realpath(__file__))

def compare_data(golden_file_lists, output_file_lists):
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        if filecmp.cmp(gold, out):
            print("PASSED!")
        else:
            print("FAILED!")
            data_same = False
    return data_same

def get_file_lists():
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists

def process():
    golden_file_lists, output_file_lists = get_file_lists()
    result = compare_data(golden_file_lists, output_file_lists)

if __name__ == '__main__':
    process()