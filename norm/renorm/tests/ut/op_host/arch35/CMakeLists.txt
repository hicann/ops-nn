# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # DISABLE_IN_OPP can keep this source out of the package's host object.
    # Add it to the test object only in that case; otherwise the registration
    # unit is already part of the host object and must not be linked twice.
    set(_renorm_tiling_source
        ${PROJECT_SOURCE_DIR}/norm/renorm/op_host/arch35/renorm_tiling_arch35.cpp)
    if(TARGET ${OPHOST_NAME}_tiling_obj)
        get_target_property(_renorm_tiling_sources ${OPHOST_NAME}_tiling_obj SOURCES)
        if(NOT _renorm_tiling_sources MATCHES "renorm_tiling_arch35\\.cpp")
            target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${_renorm_tiling_source})
        endif()
    else()
        target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${_renorm_tiling_source})
    endif()
    # add_modules_ut_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
