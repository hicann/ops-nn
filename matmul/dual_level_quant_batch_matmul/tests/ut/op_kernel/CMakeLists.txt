# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(dual_level_quant_batch_matmul "ascend950pr_9599" "-DDTYPE_X=bfloat16_t -DORIG_DTYPE_X=DT_BF16 -DFORMAT_X=FORMAT_ND \
    -DDTYPE_WEIGHT=int4b_t -DORIG_DTYPE_WEIGHT=DT_INT4 -DFORMAT_WEIGHT=FORMAT_ND \
    -DDTYPE_ANTIQUANT_SCALE=bfloat16_t -DORIG_DTYPE_ANTIQUANT_SCALE=DT_BF16 -DFORMAT_ANTIQUANT_SCALE=FORMAT_ND \
    -DDTYPE_ANTIQUANT_OFFSET=bfloat16_t -DORIG_DTYPE_ANTIQUANT_OFFSET=DT_BF16 -DFORMAT_ANTIQUANT_OFFSET=FORMAT_ND \
    -DDTYPE_QUANT_SCALE=uint64_t -DORIG_DTYPE_QUANT_SCALE=DT_UINT64 -DFORMAT_QUANT_SCALE=FORMAT_ND \
    -DDTYPE_QUANT_OFFSET=float -DORIG_DTYPE_QUANT_OFFSET=DT_FLOAT -DFORMAT_QUANT_OFFSET=FORMAT_ND \
    -DDTYPE_BIAS=float -DORIG_DTYPE_BIAS=DT_FLOAT -DFORMAT_BIAS=FORMAT_ND \
    -DDTYPE_Y=bfloat16_t -DORIG_DTYPE_Y=DT_BF16 -DFORMAT_Y=FORMAT_ND \
    -D__CCE_AICORE__=220")
endif()
