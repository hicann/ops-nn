# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(sparse4to2quant_matmul "ascend910B1" "-DDTYPE_X=int8_t -DDTYPE_SPARSE_WEIGHT=int8_t -DORIG_DTYPE_X=DT_INT8 -DORIG_DTYPE_SPARSE_WEIGHT=DT_INT8 -DDTYPE_Y=bfloat16_t -DORIG_DTYPE_Y=DT_BF16 -DDTYPE_X_SCALE=float -DORIG_DTYPE_X_SCALE=DT_BF16 -DDTYPE_BIAS=bfloat16_t -DORIG_DTYPE_BIAS=DT_BF16 -DDTYPE_SPARSE_WEIGHT_SCALE=float -DORIG_DTYPE_SPARSE_WEIGHT_SCALE=DT_FLOAT -D__CCE_AICORE__=220")
endif()