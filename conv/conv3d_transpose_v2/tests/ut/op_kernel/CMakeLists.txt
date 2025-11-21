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
    AddOpTestCase(conv3d_transpose_v2 "ascend910B1" "-DDTYPE_FILTER=bfloat16_t -DFORMAT_FILTER=FORMAT_FRACTAL_Z_3D -DDTYPE_X=bfloat16_t -DFORMAT_X=FORMAT_NDC1HWC0 -DDTYPE_Y=bfloat16_t -DFORMAT_Y=FORMAT_NDC1HWC0" "conv3d_backprop_input_v2")
endif()