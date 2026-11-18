# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#----------------------------------------------------------------------------
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    # ascend960dt 在 cann-9.3.0 tikicpulib 包中仅提供 Ascend960DT 大小写别名（tikicpulib::Ascend960DT）
    AddOpTestCase(dynamic_hi_f4_quant "Ascend960DT" "-DDTYPE_X=bfloat16_t -DDTYPE_Y=fp4x2_e1m2_t")
endif()
