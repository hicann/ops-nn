# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 脚本模式命令行入口，供 bash 等外部脚本查询本仓 CMake 侧定义的变量与函数，三种用法：
#   1. 缺省：按 socVersion 查询 npuArch（大小写不敏感，未匹配输出空串）
#        npu_arch=$(cmake -P cmake/query.cmake ascend910b 2>&1)            # npu_arch=2201
#   2. Func：按函数名调用，当前仅支持 get_npu_arch_by_soc_version/get_soc_version_by_npu_arch
#        npu_arch=$(cmake -P cmake/query.cmake Func get_npu_arch_by_soc_version ascend910b 2>&1)
#   3. Value：查询 variables.cmake 中定义的变量，列表值按行输出，bash 可直接分词
#        soc_list=($(cmake -P cmake/query.cmake Value ASCEND_ALL_COMPUTE_UNIT 2>&1))
# 注意：脚本模式 message() 输出到 stderr，bash 需用 2>&1 捕获；
#       参数缺失/函数不支持/变量未定义等错误通过非零退出码报告。
# variables.cmake 为纯变量定义（脚本模式安全，可直接 include）；
# add_library/file/execute_process 等构建期副作用位于 targets.cmake，仅由顶层 CMakeLists 引入。

if(NOT DEFINED CMAKE_SCRIPT_MODE_FILE)
  message(FATAL_ERROR "query.cmake must run in script mode: cmake -P cmake/query.cmake <command>")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/soc_arch_variables.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/func.cmake")

string(CONCAT _usage
  "usage: cmake -P cmake/query.cmake <soc_version>\n"
  "       cmake -P cmake/query.cmake Func get_npu_arch_by_soc_version <soc_version>\n"
  "       cmake -P cmake/query.cmake Value <variable_name>"
)

set(_command "${CMAKE_ARGV3}")

if("${_command}" STREQUAL "")
  # 无参数
  message(FATAL_ERROR "${_usage}")
elseif("${_command}" STREQUAL "Func")
  # 按函数名调用：query.cmake Func <func_name> <func_args...>
  set(_func "${CMAKE_ARGV4}")
  if("${_func}" STREQUAL "")
    message(FATAL_ERROR "${_usage}")
  endif()
  if("${_func}" STREQUAL "get_npu_arch_by_soc_version" AND NOT "${CMAKE_ARGV5}" STREQUAL "")
    get_npu_arch_by_soc_version("${CMAKE_ARGV5}" _arch)
    message("${_arch}")
  elseif("${_func}" STREQUAL "get_soc_version_by_npu_arch" AND NOT "${CMAKE_ARGV5}" STREQUAL "")
    get_soc_version_by_npu_arch("${CMAKE_ARGV5}" _soc)
    message("${_soc}")
  else()
    message(FATAL_ERROR "unsupported function: ${_func}\n${_usage}")
  endif()
elseif("${_command}" STREQUAL "Value")
  # 查询变量取值：query.cmake Value <variable_name>
  set(_var_name "${CMAKE_ARGV4}")
  if(${_var_name} STREQUAL "")
    message(FATAL_ERROR "${_usage}")
  elseif(NOT DEFINED "${_var_name}")
    message(FATAL_ERROR "variable is not defined: ${_var_name}\n${_usage}")
  else()
    # 列表值按行输出，便于 bash 直接分词（避免分号分隔导致的单元素数组问题）
    string(REPLACE ";" "\n" _value "${${_var_name}}")
    message("${_value}")
  endif()
else()
  # 缺省：按 socVersion 查询 npuArch（未匹配输出空串）
  get_npu_arch_by_soc_version("${_command}" _arch)
  message("${_arch}")
endif()
