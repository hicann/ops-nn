#
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

set(CMAKE_LIBRARY_OUTPUT_DIRECTOR ${PROJECT_SOURCE_DIR}/build)

file(GLOB_RECURSE OP_API_FILES "*.cpp")

add_library(opapi_math SHARED ${OP_API_FILES})

target_compile_definitions(opapi_math PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    LOG_CPP
)

target_compile_options(opapi_math PRIVATE -Wno-unused-parameter)

target_include_directories(opapi_math PRIVATE
    ${OPAPI_INCLUDE}
)

if(ENABLE_TEST)
    add_library(opapi_math_stub SHARED ${OP_API_FILES})

    target_compile_definitions(opapi_math_stub PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    target_compile_options(opapi_math_stub PRIVATE -Wno-unused-parameter)

    target_include_directories(opapi_math_stub PRIVATE
        ${OPAPI_INCLUDE}
    )
endif()