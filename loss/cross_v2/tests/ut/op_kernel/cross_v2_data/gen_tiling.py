# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
import numpy as np
import sys

def main(tiling_np):
    tiling_data = np.array(
        tiling_np, dtype=np.uint64)
    tiling_file = open("tiling.bin", "wb")
    tiling_data.tofile(tiling_file)

if __name__ == '__main__':
    case_name = sys.argv[1]
    case_list = {
        "test_case_float_0" : [1,40,1,8,8],
        "test_case_float_1" : [320,40,40,8,8],
    }
    tiling_np = case_list.get(case_name)
    main(tiling_np)