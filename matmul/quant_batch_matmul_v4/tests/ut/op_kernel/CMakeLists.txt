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
    AddOpTestCase(quant_batch_matmul_v4 "ascend910B1" "-DKFC_L1_RESERVER_SIZE=0 -DDTYPE_X1=int8_t -DDTYPE_X2=int8_t -DDTYPE_BIAS=float32 -DDTYPE_Y=bfloat16_t -DFORMAT_X1=2 -DFORMAT_X2=2" quant_batch_matmul_v3)
endif()