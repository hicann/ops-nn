# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# param1: 算子名（kernel 命名）
# param2: soc 版本（arch35 / RegBase → ascend950pr_9599）
# param3: 编译宏。grads 的 dtype 由构建系统按输入名注入 DTYPE_GRADS；kernel 入口按此实例化。
#         默认组合取 float（FP32 直算路径），FP16/BF16 路径在测试内部另行覆盖。
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(bn_infer_grad "ascend950pr_9599" "-DDTYPE_GRADS=float")
endif()
