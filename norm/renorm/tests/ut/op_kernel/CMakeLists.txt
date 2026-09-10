# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(renorm "ascend950pr_9599" "-DDTYPE_Y_BACKPROP=float -DDTYPE_SCALE=float")
    # A5 ops-info names the generated entry renorm_apt.cpp, while this UT is
    # intentionally kept at the stable test_renorm.cpp path.  Add the test
    # source explicitly so it is compiled as an object in the kernel target.
    if(TARGET opkernel_renorm)
        target_sources(opkernel_renorm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_renorm.cpp)
    endif()
endif()
