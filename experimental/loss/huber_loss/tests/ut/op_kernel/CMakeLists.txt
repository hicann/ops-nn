# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# The tiling and infershape sources are compiled in so the kernel UT links
# against the real host side rather than a stub.
if((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    set(huber_loss_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/huber_loss_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/huber_loss_infershape.cpp
    )
    AddOpTestCase(huber_loss "ascend910B1"
                  "-DDTYPE_INPUT=float -DDTYPE_TARGET=float -DDTYPE_LOSS=float"
                  "${huber_loss_tiling_files}")
endif()
