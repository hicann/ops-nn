# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 算子类别清单
set(OP_CATEGORY_LIST "activation" "conv" "foreach" "vfusion" "index" "loss" "matmul" "norm" "optim" "pooling" "quant" "rnn" "control")

set(COMMON_NAME common_${PKG_NAME})
set(OPHOST_NAME ophost_${PKG_NAME})
set(OPAPI_NAME opapi_${PKG_NAME})
set(OPGRAPH_NAME opgraph_${PKG_NAME})
set(GRAPH_PLUGIN_NAME graph_plugin_${PKG_NAME})
set(VENDOR_PACKAGE_NAME ${VENDOR_NAME}_nn)

if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/build/third_party)
endif()
if(NOT CANN_3RD_PKG_PATH)
  set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/build/third_party/pkg)
endif()

# interface, 用于收集aclnn/aclnn_inner/aclnn_exclude的def文件
add_library(${OPHOST_NAME}_opdef_aclnn_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE)
add_library(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE)
# interface, 用于收集ops proto头文件
add_library(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE)

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
  set(IMPL_INSTALL_DIR                packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/ascendc)
  set(IMPL_DYNAMIC_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic)
  set(BIN_KERNEL_INSTALL_DIR          packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/kernel)
  set(BIN_KERNEL_CONFIG_INSTALL_DIR   packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/kernel/config)
  set(OPTILING_INSTALL_DIR            packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/op_tiling/)
  set(OPTILING_LIB_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(OPPROTO_INC_INSTALL_DIR         packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto/inc)
  set(OPPROTO_LIB_INSTALL_DIR         packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(CUST_AICPU_KERNEL_IMPL          packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/cpu/aicpu_kernel/impl)
  set(CUST_AICPU_KERNEL_CONFIG        packages/vendors/${VENDOR_PACKAGE_NAME}/op_impl/cpu/config)
  set(CUST_AICPU_OP_PROTO             packages/vendors/${VENDOR_PACKAGE_NAME}/op_proto)
  set(VERSION_INFO_INSTALL_DIR        packages/vendors/${VENDOR_PACKAGE_NAME}/)
else()
  # built-in package install path
  set(ACLNN_INC_INSTALL_DIR           ops_nn/built-in/op_api/include)
  set(ACLNN_OP_INC_INSTALL_DIR        ops_nn/built-in/op_api/include/aclnnop)
  set(ACLNN_LIB_INSTALL_DIR           ops_nn/built-in/op_api/lib64/${CMAKE_SYSTEM_PROCESSOR})
  set(OPS_INFO_INSTALL_DIR            ops_nn/built-in/op_impl/ai_core/tbe/config)
  set(IMPL_INSTALL_DIR                ops_nn/built-in/op_impl/ai_core/tbe/impl/ascendc)
  set(IMPL_DYNAMIC_INSTALL_DIR        ops_nn/built-in/op_impl/ai_core/tbe/impl/dynamic)
  set(BIN_KERNEL_INSTALL_DIR          ops_nn/built-in/op_impl/ai_core/tbe/kernel)
  set(BIN_KERNEL_CONFIG_INSTALL_DIR   ops_nn/built-in/op_impl/ai_core/tbe/kernel/config)
  set(OPHOST_INC_INSTALL_PATH         ops_nn/built-in/op_impl/ai_core/tbe/op_host/include)
  set(OPHOST_LIB_INSTALL_PATH         ops_nn/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(OPTILING_LIB_INSTALL_DIR        ${OPHOST_LIB_INSTALL_PATH})
  set(OPGRAPH_INC_INSTALL_DIR         ops_nn/built-in/op_graph/inc)
  set(OPGRAPH_LIB_INSTALL_DIR         ops_nn/built-in/op_graph/lib/${CMAKE_SYSTEM_PROCESSOR})
  set(COMMON_INC_INSTALL_DIR          ops_nn/include)
  set(COMMON_LIB_INSTALL_DIR          ops_nn/lib)
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
  ${OPS_NN_DIR}/common/stub/op_tiling
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
