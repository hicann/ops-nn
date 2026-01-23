 # -----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 # -----------------------------------------------------------------------------
 
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(quant_batch_matmul_v3 "ascend910B1" "-DDTYPE_X1=int8_t -DDTYPE_X2=int8_t -DORIG_DTYPE_X1=DT_INT8 -DORIG_DTYPE_X2=DT_INT8 -DFORMAT_X1=2 -DFORMAT_X2=2 -DDTYPE_Y=bfloat16_t -DORIG_DTYPE_Y=DT_BF16 -DDTYPE_SCALE=bfloat16_t -DORIG_DTYPE_SCALE=DT_BF16 -DDTYPE_BIAS=bfloat16_t -DORIG_DTYPE_BIAS=DT_BF16 -DDTYPE_PERTOKEN_SCALE=float -DORIG_DTYPE_PERTOKEN_SCALE=DT_FLOAT -D__CCE_AICORE__=220")
    AddOpTestCase(quant_batch_matmul_v3 "ascend910_9599" "" "MULTI_KERNEL_TARGET")
endif()