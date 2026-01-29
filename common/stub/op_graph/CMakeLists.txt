# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

set(CMAKE_LIBRARY_OUTPUT_DIRECTOR ${PROJECT_SOURCE_DIR}/build)

file(GLOB_RECURSE OP_PROTO_FILES "*.cpp")

add_library(proto_math SHARED ${OP_PROTO_FILES})

target_compile_definitions(proto_math PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    LOG_CPP
)

target_link_libraries(
        proto_math
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                -Wl,--no-as-needed
                register
                $<$<TARGET_EXISTS:opsbase>:opsbase>
                -Wl,--as-needed
                -Wl,--whole-archive
                rt2_registry_static
                -Wl,--no-whole-archive
        )
 # 生成 es_math 
  add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET proto_math
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_packages_math
  )
  
