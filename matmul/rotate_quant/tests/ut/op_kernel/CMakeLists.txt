# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# v121(ascend960dt) apt kernel UT: 按 dtype 组合注册, 用例位于 arch35/ 子目录
# (与根目录的 arch22(910B) kernel UT 隔离, 避免被根 CMakeLists 的 AddOpTestCase("ascend910B1") GLOB 误收集)
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    # hif4 output (y=hifloat4x2_t, scale=float for HIFLOAT4_SCALE)
    AddOpTestCase(rotate_quant "ascend960dt"
                  "-DDTYPE_X=bfloat16_t -DDTYPE_Y=hifloat4x2_t -DDTYPE_SCALE=float"
                  UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
    AddOpTestCase(rotate_quant "ascend960dt"
                  "-DDTYPE_X=half -DDTYPE_Y=hifloat4x2_t -DDTYPE_SCALE=float"
                  UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
    # mxfp4 output (y=fp4x2_e2m1_t, scale=fp8_e8m0_t)
    AddOpTestCase(rotate_quant "ascend960dt"
                  "-DDTYPE_X=bfloat16_t -DDTYPE_Y=fp4x2_e2m1_t -DDTYPE_SCALE=fp8_e8m0_t"
                  UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
    AddOpTestCase(rotate_quant "ascend960dt"
                  "-DDTYPE_X=half -DDTYPE_Y=fp4x2_e2m1_t -DDTYPE_SCALE=fp8_e8m0_t"
                  UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
    # mxfp8 output (y=fp8_e4m3fn_t / fp8_e5m2_t, scale=fp8_e8m0_t)
    AddOpTestCase(rotate_quant "ascend960dt"
                  "-DDTYPE_X=bfloat16_t -DDTYPE_Y=fp8_e4m3fn_t -DDTYPE_SCALE=fp8_e8m0_t"
                  UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
    AddOpTestCase(rotate_quant "ascend960dt"
                  "-DDTYPE_X=half -DDTYPE_Y=fp8_e5m2_t -DDTYPE_SCALE=fp8_e8m0_t"
                  UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
endif()
