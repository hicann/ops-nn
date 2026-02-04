# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 算子类别清单
set(OP_CATEGORY_LIST "matmul" "conv" "reliability" "activation" "foreach" "hash" "vfusion" "index" "loss" "norm" "optim" "pooling" "quant" "rnn" "control")

set(COMMON_NAME common_${PKG_NAME})
set(OPHOST_NAME ophost_${PKG_NAME})
set(OPSTATIC_NAME cann_${PKG_NAME}_static)
set(OPAPI_NAME opapi_${PKG_NAME})
set(OPGRAPH_NAME opgraph_${PKG_NAME})
set(ONNX_PLUGIN_NAME op_${PKG_NAME}_onnx_plugin)
set(GRAPH_PLUGIN_NAME graph_plugin_${PKG_NAME})
set(VENDOR_PACKAGE_NAME ${VENDOR_NAME}_nn)

set(ACLNN_EXTRA_HEADERS "" CACHE STRING "Aclnn Extra Headers" FORCE)
set(ACLNN_EXTRA_SRCS "" CACHE STRING "Aclnn Extra Sources" FORCE)

if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/build/third_party)
endif()
if(NOT CANN_3RD_PKG_PATH)
  set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/build/third_party/pkg)
endif()

message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(ARCH x86_64)
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64|arm64|arm")
  set(ARCH aarch64)
else()
  message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# interface, 用于收集aclnn/aclnn_inner/aclnn_exclude的def文件
add_library(${OPHOST_NAME}_opdef_aclnn_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE)
add_library(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE)
# interface, 用于收集ops proto头文件
add_library(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE)

set(OP_GRAPH_MODULE_NAME ${PKG_NAME}_op_graph_ut)

# global variables
set(COMPILED_OPS CACHE STRING "Compiled Ops" FORCE)
set(COMPILED_OP_DIRS CACHE STRING "Compiled Ops Dirs" FORCE)

# src path
get_filename_component(OPS_NN_CMAKE_DIR           "${OPS_NN_DIR}/cmake"                               REALPATH)
get_filename_component(OPS_NN_COMMON_INC          "${OPS_NN_DIR}/common/inc"                      REALPATH)
get_filename_component(OPS_NN_COMMON_INC_COMMON   "${OPS_NN_COMMON_INC}/common"                       REALPATH)
get_filename_component(OPS_NN_COMMON_INC_EXTERNAL "${OPS_NN_COMMON_INC}/external"                     REALPATH)
get_filename_component(OPS_NN_COMMON_INC_HEADERS  "${OPS_NN_COMMON_INC_EXTERNAL}/aclnn_kernels"       REALPATH)
get_filename_component(OPS_KERNEL_BINARY_SCRIPT     "${OPS_NN_DIR}/scripts/kernel/binary_script"       REALPATH)
get_filename_component(OPS_KERNEL_BINARY_CONFIG     "${OPS_NN_DIR}/scripts/kernel/binary_config"       REALPATH)
get_filename_component(OPS_CMCT               "${OPS_NN_DIR}/matmul/common/cmct"           REALPATH)

# python
if(NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
  set(ASCEND_PYTHON_EXECUTABLE python3 CACHE STRING "")
endif()

# install path
if(ENABLE_CUSTOM)
  # custom package install path
  set(ACLNN_INC_INSTALL_DIR           packages/vendors/${VENDOR_PACKAGE_NAME}/op_api/include)
  set(ACLNN_OP_INC_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/op_api/include/aclnnop)
  set(ACLNN_LIB_INSTALL_DIR           packages/vendors/${VENDOR_PACKAGE_NAME}/op_api/lib)
  set(OPS_INFO_INSTALL_DIR            packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/config)
  set(IMPL_INSTALL_DIR                packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/${VENDOR_PACKAGE_NAME}_impl/ascendc)
  set(IMPL_DYNAMIC_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/${VENDOR_PACKAGE_NAME}_impl/dynamic)
  set(BIN_KERNEL_INSTALL_DIR          packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/kernel)
  set(BIN_KERNEL_CONFIG_INSTALL_DIR   packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/kernel/config)
  set(OPTILING_INSTALL_DIR            packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/op_tiling/)
  set(OPTILING_LIB_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(OPPROTO_INC_INSTALL_DIR         packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto/inc)
  set(OPPROTO_LIB_INSTALL_DIR         packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(AICPU_KERNEL_IMPL               packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/cpu/aicpu_kernel/impl)
  set(AICPU_JSON_CONFIG               packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/cpu/config)
  set(CUST_AICPU_OP_PROTO             packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto)
  set(ES_INC_INSTALL_DIR              packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto/es/include)
 	set(ES_LIB_INSTALL_DIR              packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto/es/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(VERSION_INFO_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/)
  set(PACK_CUSTOM_NAME                "cann-ops-nn-${VENDOR_NAME}-linux.${ARCH}")
else()
  # built-in package install path
  set(ACLNN_INC_INSTALL_DIR           opp/include/aclnnop)
  set(ACLNN_OP_INC_INSTALL_DIR        opp/include/aclnnop/level2)
  set(ACLNN_LIB_INSTALL_DIR           opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(OPS_INFO_INSTALL_DIR            opp/built-in/op_impl/ai_core/tbe/config)
  set(IMPL_INSTALL_DIR                opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc)
  set(IMPL_DYNAMIC_INSTALL_DIR        opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/dynamic)
  set(BIN_KERNEL_INSTALL_DIR          opp/built-in/op_impl/ai_core/tbe/kernel)
  set(BIN_KERNEL_CONFIG_INSTALL_DIR   opp/built-in/op_impl/ai_core/tbe/kernel/config)
  set(OPHOST_LIB_INSTALL_PATH         opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(AICPU_KERNEL_IMPL               opp/built-in/op_impl/aicpu/kernel)
  set(AICPU_JSON_CONFIG               opp/built-in/op_impl/aicpu/config)
  set(OPTILING_LIB_INSTALL_DIR        ${OPHOST_LIB_INSTALL_PATH})
  set(OPGRAPH_INC_INSTALL_DIR         opp/built-in/op_graph/inc)
  set(OPGRAPH_LIB_INSTALL_DIR         opp/built-in/op_graph/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(ONNX_PLUGIN_LIB_INSTALL_DIR     opp/built-in/framework/onnx)
  set(VERSION_INFO_INSTALL_DIR        ops_nn)
endif()

# util path
set(ASCEND_TENSOR_COMPILER_PATH ${ASCEND_DIR}/compiler)
set(ASCEND_CCEC_COMPILER_PATH ${ASCEND_TENSOR_COMPILER_PATH}/ccec_compiler/bin)
set(OP_BUILD_TOOL ${ASCEND_DIR}/tools/opbuild/op_build)

# tmp path
set(ASCEND_TMP_PATH ${CMAKE_BINARY_DIR}/tmp)
set(ASCEND_SUB_CONFIG_PATH ${ASCEND_TMP_PATH}/ops_config.txt)
file(MAKE_DIRECTORY ${ASCEND_TMP_PATH})
file(REMOVE ${ASCEND_SUB_CONFIG_PATH})
set(UT_PATH ${PROJECT_SOURCE_DIR}/tests/ut)

# output path
set(ASCEND_AUTOGEN_PATH     ${CMAKE_BINARY_DIR}/autogen)
set(ASCEND_KERNEL_SRC_DST   ${CMAKE_BINARY_DIR}/tbe/ascendc)
set(ASCEND_KERNEL_CONF_DST  ${CMAKE_BINARY_DIR}/tbe/config)
set(ASCEND_GRAPH_CONF_DST   ${CMAKE_BINARY_DIR}/tbe/graph)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
file(MAKE_DIRECTORY ${ASCEND_KERNEL_SRC_DST})
file(MAKE_DIRECTORY ${ASCEND_KERNEL_CONF_DST})
file(MAKE_DIRECTORY ${ASCEND_GRAPH_CONF_DST})
set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
set(CUSTOM_OPC_OPTIONS "custom_opc_options.ini")
execute_process(
  COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
  COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
  COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
  COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
)

# pack path
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build_out)

set(OPAPI_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${OPBASE_INC_DIRS}
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/profiling # include profiling/prof_common.h
  ${METADEF_INCLUDE_DIRS}
  ${NNOPBASE_INCLUDE_DIRS}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${JSON_INCLUDE}
  ${OPS_NN_DIR}
  ${AICPU_INC_DIRS}
  ${OPS_NN_DIR}/common/inc
  ${OPS_NN_DIR}/common/inc/op_api
  ${TOP_DIR}/output/${PRODUCT}/aclnnop_resource
  ${OPS_NN_DIR}/common/stub/op_api
  ${ASCEND_AUTOGEN_PATH}/inner
)

set(OP_TILING_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${JSON_INCLUDE}
  ${OPBASE_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${TILINGAPI_INC_DIRS}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${OPS_NN_DIR}
  ${OPS_NN_DIR}/common/inc
  ${OPS_NN_DIR}/common/inc/op_host
)

set(OP_PROTO_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${OPBASE_INC_DIRS}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${OPS_NN_DIR}
  ${OPS_NN_DIR}/matmul
  ${OPS_NN_DIR}/conv
  ${OPS_NN_DIR}/vfusion
  ${OPS_NN_DIR}/common/inc
)

set(AICPU_INCLUDE
  ${OPBASE_INC_DIRS}
  ${AICPU_INC_DIRS}
  ${C_SEC_INCLUDE}
  ${NNOPBASE_INCLUDE_DIRS}
  ${HCCL_EXTERNAL_INCLUDE}
  ${OPS_NN_DIR}/common/include/common
  ${METADEF_INCLUDE_DIRS}
)

set(ONNX_PLUGIN_COMMON_INCLUDE
  ${OPS_NN_DIR}
  ${OPS_NN_DIR}/common/inc/framework
  ${OPS_NN_DIR}/common/stub/inc/framework
  ${OPS_NN_DIR}/common/inc/op_graph
)

set(AICPU_DEFINITIONS
  -O2
  -std=c++14
  -fstack-protector-all
  -fvisibility-inlines-hidden
  -fvisibility=hidden
  -frename-registers
  -fpeel-loops
  -DEIGEN_NO_DEBUG
  -DEIGEN_MPL2_ONLY
  -DNDEBUG
  -DEIGEN_HAS_CXX11_MATH
  -DEIGEN_OS_GNULINUX
  -DEigen=ascend_Eigen
  -fno-common
  -fPIC
)
