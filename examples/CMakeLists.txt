# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
# 遍历子目录
foreach(SUBDIR ${SUBDIRECTORIES})
  # 检查是否跳过目录
  if("${SUBDIR}" STREQUAL "fast_kernel_launch_example")
    continue()
  endif()

  # PyPTO 示例算子依赖 configure 阶段 codegen 生成的 tiling/tilingkey 头文件（enable_pypto_kernel
  # 按 ASCEND_OP_NAME 精确匹配执行）；未被 --ops 指定编译时不得加载，否则其 tiling 源会被加入
  # 共享 ophost_nn_tiling_obj 目标但生成头文件缺失，导致 --ops=add_example 等既有构建失败
  if("${SUBDIR}" STREQUAL "add_example_pypto")
    list(FIND ASCEND_COMPILE_OPS ${SUBDIR} _pypto_index)
    if(_pypto_index EQUAL -1)
      continue()
    endif()
  endif()

  # 检查子目录中是否存在 CMakeLists.txt
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
    add_subdirectory(${SUBDIR})
  endif()
endforeach()
