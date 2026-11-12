# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_modules_sources(
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE apply_ftrl
    ACLNNTYPE aclnn_inner
)

add_kernel_sources(
    KERNEL_SRC arch35/apply_ftrl.cpp
    COMPUTE_UNITS ascend950 ascend350 ascend960dt ascend960pr
    AUTO_SYNC false
    OPTIONS "-ffp-contract=off"
)

if((ENABLE_TEST OR BENCHMARK) AND NOT OP_ONLY_COMPILE AND (UT_TEST_ALL OR OP_HOST_UT))
    add_modules_ut_sources(
        HOSTNAME ${OP_TILING_MODULE_NAME}
        MODE PRIVATE
        DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_host
    )
    add_modules_ut_sources(
        HOSTNAME ${OP_INFERSHAPE_MODULE_NAME}
        MODE PRIVATE
        DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_host
    )
    # EZ1008/EZ1014 are registered by nnopbase; keep its registration initializer in host UTs.
    foreach(module ${OP_TILING_MODULE_NAME} ${OP_INFERSHAPE_MODULE_NAME})
        if(TARGET ${module}_static_lib)
            # Keep flags and library together: CMake may deduplicate standalone --no-as-needed flags.
            target_link_libraries(${module}_static_lib PRIVATE
                "-Wl$<COMMA>--push-state$<COMMA>--no-as-needed$<COMMA>$<TARGET_FILE:nnopbase>$<COMMA>--pop-state")
        endif()
    endforeach()
endif()
