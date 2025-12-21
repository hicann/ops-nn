# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
import sys
import numpy as np

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float32)
    golden = np.fromfile(golden, dtype=np.float32)
    return np.allclose(real_result, golden, 1e-4, 1e-4)


if __name__ == '__main__':
    if verify_result(sys.argv[1], sys.argv[2]):
        exit(0)
    else:
        exit(-1)
