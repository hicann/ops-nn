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
    if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend950pr_9599" IN_LIST ASCEND_COMPUTE_UNIT)
        AddOpTestCase(apply_adam_w_quant "ascend950pr_9599" "-DORIG_DTYPE_VAR=DT_FLOAT16 -D__CCE_AICORE__=350")
    else()
        AddOpTestCase(apply_adam_w_quant "ascend910B1" "-DORIG_DTYPE_VAR=DT_FLOAT16")
    endif()
endif()
