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
    AddOpTestCase(relu_v3 "ascend910B1" "-DDTYPE_X=float -DDTYPE_Y=float -DDTYPE_MASK=uint8_t" "${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/relu_v3_tiling.cpp")
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/compare_data.py ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/gen_data.py COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/relu_v3_data/compare_data.py ${CMAKE_CURRENT_SOURCE_DIR}/relu_v3_data/gen_data.py ${CMAKE_BINARY_DIR}/tests/ut/op_kernel DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/relu_v3_data/compare_data.py ${CMAKE_CURRENT_SOURCE_DIR}/relu_v3_data/gen_data.py)
    add_custom_target(relu_v3_Ascend910B1_sync_py DEPENDS ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/compare_data.py ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/gen_data.py)
    add_dependencies(opkernel_relu_v3 relu_v3_Ascend910B1_sync_py)
endif ()
