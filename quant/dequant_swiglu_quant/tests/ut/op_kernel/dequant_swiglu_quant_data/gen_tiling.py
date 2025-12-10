# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys


"""
struct DequantSwigluQuantBaseTilingData {
  int64_t inDimx;
  int64_t inDimy;
  int64_t outDimy;
  int64_t UbFactorDimx;
  int64_t UbFactorDimy;
  int64_t usedCoreNum;
  int64_t maxCoreNum;
  int64_t inGroupNum;
  int64_t quantMode;
  int64_t actRight;
  int64_t quantScaleDtype;
  int64_t groupIndexDtype;
  int64_t needSmoothScale;
  int64_t speGroupType;
  int64_t activationScaleIsEmpty;
  int64_t quantIsOne;
  int64_t swigluMode;
  float clampLimit;
  float gluAlpha;
  float gluBias;
};
"""

params_info = {
    "test_dequant_swiglu_quant_1": [128, 2048, 1024, 4, 1024, 32, 32, 1, 1, 0, 4, 8, 1, 0, 0, 0, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_2": [128, 2048, 1024, 4, 1024, 32, 32, 1, 1, 0, 4, 8, 1, 0, 0, 0, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_3": [128, 2048, 1024, 4, 1024, 32, 32, 1, 1, 0, 4, 8, 1, 0, 0, 0, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_4": [128, 2048, 1024, 4, 1024, 32, 32, 1, 1, 0, 4, 8, 1, 0, 0, 0, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_5": [128, 2048, 1024, 4, 1024, 32, 32, 1, 1, 0, 4, 8, 1, 0, 1, 0, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_6": [128, 2048, 1024, 4, 1024, 32, 32, 1, 0, 0, 4, 8, 1, 0, 1, 0, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_7": [128, 2048, 1024, 4, 1024, 32, 32, 1, 0, 0, 4, 8, 1, 0, 1, 1, 0, 0, 0, 0],
    "test_dequant_swiglu_quant_bias_and_swiglugate": [128, 2048, 1024, 4, 1024, 32, 32, 1, 0, 0, 4, 8, 1, 0, 0, 0, 1, 7.0, 1.702, 1.0],
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()

