# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    # 算子自己的tiling文件路径
    set(sync_batch_norm_backward_reduce_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/sync_batch_norm_backward_reduce_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/sync_batch_norm_backward_reduce_infershape.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："ascend910_95;ascend910b"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(sync_batch_norm_backward_reduce "ascend910B1" "" "${sync_batch_norm_backward_reduce_tiling_files}")
endif()
