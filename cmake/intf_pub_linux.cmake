# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# intf_pub for c++11
if(NOT TARGET intf_pub)
  add_library(intf_pub INTERFACE)
  target_compile_options(intf_pub INTERFACE
    -Wall
    -fPIC
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
  )
  target_compile_definitions(intf_pub INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  )
  target_link_options(intf_pub INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none>
  )
  target_link_directories(intf_pub INTERFACE)
  target_link_libraries(intf_pub INTERFACE
    -lpthread
  )
endif()

# intf_pub_cxx14 for c++14
if(NOT TARGET intf_pub_cxx14)
  add_library(intf_pub_cxx14 INTERFACE)
  target_compile_options(intf_pub_cxx14 INTERFACE
    -Wall
    -fPIC
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
  )
  target_compile_definitions(intf_pub_cxx14 INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  )
  target_link_options(intf_pub_cxx14 INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none>
  )
  target_link_directories(intf_pub_cxx14 INTERFACE)
  target_link_libraries(intf_pub_cxx14 INTERFACE
    -lpthread
  )
endif()

# intf_pub_cxx14 for c++17
if(NOT TARGET intf_pub_cxx17)
  add_library(intf_pub_cxx17 INTERFACE)
  target_compile_options(intf_pub_cxx17 INTERFACE
      -Wall
      -fPIC
      $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
      $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
  )
  target_compile_definitions(intf_pub_cxx17 INTERFACE
      _GLIBCXX_USE_CXX11_ABI=0
      $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
      $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  )
  target_link_options(intf_pub_cxx17 INTERFACE
      -Wl,-z,relro
      -Wl,-z,now
      -Wl,-z,noexecstack
      $<$<CONFIG:Release>:-Wl,--build-id=none>)
  target_link_directories(intf_pub_cxx17 INTERFACE)
  target_link_libraries(intf_pub_cxx17 INTERFACE
    -lpthread
  )
endif()
