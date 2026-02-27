# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ophost shared
function(gen_ophost_symbol)
  if (NOT TARGET ${OPHOST_NAME}_infer_obj AND NOT TARGET ${OPHOST_NAME}_tiling_obj AND NOT TARGET ${OPHOST_NAME}_aicpu_objs)
    return()
  endif()
  add_library(
    ${OPHOST_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
    $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
    )

  target_link_libraries(
    ${OPHOST_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            tiling_api
            -Wl,--no-whole-archive
            -Wl,-Bsymbolic
            unified_dlog
    )

  target_link_directories(${OPHOST_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  install(
    TARGETS ${OPHOST_NAME}
    LIBRARY DESTINATION ${OPHOST_LIB_INSTALL_PATH}
    )
endfunction()

# gen es_nn
function(gen_es_nn_lib_ready)
  # 合并proto.h生成ops_proto_nn.h和ops_proto_nn.cpp 
  merge_graph_headers(TARGET merge_ops_proto_${PKG_NAME} OUT_DIR ${ASCEND_GRAPH_CONF_DST})
  add_library(
    proto_${PKG_NAME} SHARED
    ${ASCEND_GRAPH_CONF_DST}/ops_proto_nn.cpp
  )
  add_dependencies(proto_${PKG_NAME} merge_ops_proto_${PKG_NAME})
  target_link_libraries(
    proto_${PKG_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
      c_sec
      -Wl,--no-as-needed
      register
      -Wl,--as-needed
    )
  target_link_directories(proto_${PKG_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
  
  # 生成 es_nn 
  add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_${PKG_NAME}
    OPP_PROTO_TARGET proto_${PKG_NAME}
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_packages
  )
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/es_packages
    DESTINATION ${VERSION_INFO_INSTALL_DIR}
    OPTIONAL
  )
endfunction()

# gen es_nn for custom
function(gen_es_nn_lib_ready_cust)
  # 合并proto.h生成ops_proto_nn.h和ops_proto_nn.cpp 
  merge_graph_headers(TARGET merge_ops_proto_${PKG_NAME}_cust OUT_DIR ${ASCEND_GRAPH_CONF_DST})
  add_library(
    proto_${PKG_NAME}_cust SHARED
    ${ASCEND_GRAPH_CONF_DST}/ops_proto_nn.cpp
  )
  add_dependencies(proto_${PKG_NAME}_cust merge_ops_proto_${PKG_NAME}_cust)
  target_link_libraries(
    proto_${PKG_NAME}_cust
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
      c_sec
      -Wl,--no-as-needed
      register
      -Wl,--as-needed
    )
  target_link_directories(proto_${PKG_NAME}_cust PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
  
  # 生成 es_nn 
  add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_${PKG_NAME}
    OPP_PROTO_TARGET proto_${PKG_NAME}_cust
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_packages
  )
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/es_packages/include/es_${PKG_NAME}/
    DESTINATION ${ES_INC_INSTALL_DIR}
    OPTIONAL
  )
  install(
    FILES ${CMAKE_BINARY_DIR}/es_packages/lib64/libes_${PKG_NAME}.so
    DESTINATION ${ES_LIB_INSTALL_DIR}
    OPTIONAL
  )
endfunction()

# graph_plugin shared
function(gen_opgraph_symbol)
  gen_es_nn_lib_ready()

  if(TARGET ${GRAPH_PLUGIN_NAME}_obj)
    unset(GRAPH_SOURCE)
    get_target_property(GRAPH_SOURCE ${GRAPH_PLUGIN_NAME}_obj SOURCES)
    if(GRAPH_SOURCE)
      add_dependencies(${GRAPH_PLUGIN_NAME}_obj
        build_es_math
        build_es_nn
      )
      target_link_libraries(${GRAPH_PLUGIN_NAME}_obj
        PRIVATE
        es_math
        es_nn
      )
      add_library(
        ${OPGRAPH_NAME} SHARED
        $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
        $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
        $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      )
      
      target_link_libraries(
        ${OPGRAPH_NAME}
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                -Wl,--no-as-needed
                register
                -Wl,--as-needed
                -Wl,--whole-archive
                rt2_registry_static
                -Wl,--no-whole-archive
                -Wl,-Bsymbolic
                ge_compiler
                es_math
                es_nn
                unified_dlog
                ascendalog
        )

      target_link_directories(${OPGRAPH_NAME} PRIVATE 
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
        ${CMAKE_BINARY_DIR}/es_packages/lib64
      )
      set_target_properties(${OPGRAPH_NAME} PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/opp/built-in/op_proto
      )
      install(
        TARGETS ${OPGRAPH_NAME}
        LIBRARY DESTINATION ${OPGRAPH_LIB_INSTALL_DIR}
      )
    endif()
  endif()
  install(
    FILES ${ASCEND_GRAPH_CONF_DST}/ops_proto_nn.h
    DESTINATION ${OPGRAPH_INC_INSTALL_DIR}
    OPTIONAL
    )

endfunction()

function(gen_opapi_symbol)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj AND NOT TARGET opbuild_gen_aclnn_all)
    return()
  endif()
  # opapi shared
  add_library(
    ${OPAPI_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
    $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )

  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    add_dependencies(${OPAPI_NAME} opapi_math)
  endif()

  target_link_libraries(
    ${OPAPI_NAME}
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    PRIVATE c_sec nnopbase $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:opapi_math>>
    )

  install(
    TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
    )
endfunction()

function(gen_cust_opapi_symbol)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj AND NOT TARGET opbuild_gen_aclnn_all)
    return()
  endif()
  # op_api
  npu_op_library(cust_opapi ACLNN)

  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    add_dependencies(cust_opapi opapi_math)
  endif()

  target_sources(
    cust_opapi
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
           $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )
  target_link_libraries(
    cust_opapi
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    PRIVATE $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:opapi_math>>
    )
endfunction()

function(gen_cust_optiling_symbol)
  # op_tiling
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    return()
  endif()
  npu_op_library(cust_opmaster TILING)
  target_sources(
    cust_opmaster
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
           $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
           $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
           $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
    )

  target_link_libraries(
    cust_opmaster
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    -Wl,-Bsymbolic
    )
endfunction()

function(gen_cust_proto_symbol)
  # op_proto
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    return()
  endif()
  npu_op_library(cust_proto GRAPH)

  set(NEED_LINK_ES OFF)
  if(TARGET ${GRAPH_PLUGIN_NAME}_obj)
    unset(GRAPH_SOURCE)
    get_target_property(GRAPH_SOURCE ${GRAPH_PLUGIN_NAME}_obj SOURCES)
    if(GRAPH_SOURCE)
      # 添加obj依赖es
      gen_es_nn_lib_ready_cust()
      add_dependencies(${GRAPH_PLUGIN_NAME}_obj
        build_es_math
        build_es_nn
      )
      target_link_libraries(${GRAPH_PLUGIN_NAME}_obj
        PRIVATE
        es_math
        es_nn
      )
      set(NEED_LINK_ES ON)
    endif()
  endif()
  
  target_sources(
    cust_proto
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
           $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
           $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
           $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
    )
  
  target_link_libraries(
    cust_proto
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    ge_compiler
    )

  if(NEED_LINK_ES)
    add_dependencies(cust_proto build_es_math build_es_nn)

    target_link_directories(cust_proto
      PRIVATE
        ${CMAKE_BINARY_DIR}/es_packages/lib64
        ${ES_LIB_INSTALL_DIR}
    )
    target_link_libraries(cust_proto
      PRIVATE
        -Wl,--no-as-needed
        es_math
        es_nn
        -Wl,--as-needed
    )
  endif()
  file(GLOB_RECURSE proto_headers ${ASCEND_AUTOGEN_PATH}/*_proto.h)
  install(
    FILES ${proto_headers}
    DESTINATION ${OPPROTO_INC_INSTALL_DIR}
    OPTIONAL
    )
endfunction()

function(gen_aicpu_json_symbol enable_built_in)
  get_property(ALL_AICPU_JSON_FILES GLOBAL PROPERTY AICPU_JSON_FILES)
  if(NOT ALL_AICPU_JSON_FILES)
    message(STATUS "No aicpu json files to merge, skipping.")
    return()
  endif()

  set(MERGED_JSON ${CMAKE_BINARY_DIR}/cust_aicpu_kernel.json)
  if(enable_built_in)
    set(MERGED_JSON ${CMAKE_BINARY_DIR}/aicpu_nn.json)
  endif()

  add_custom_command(
    OUTPUT ${MERGED_JSON}
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/util/merge_aicpu_info_json.sh ${CMAKE_SOURCE_DIR} ${MERGED_JSON} ${ALL_AICPU_JSON_FILES}
    DEPENDS ${ALL_AICPU_JSON_FILES}
    COMMENT "Merging Json files into ${MERGED_JSON}"
    VERBATIM
  )
  add_custom_target(merge_aicpu_json ALL DEPENDS ${MERGED_JSON})
  install(
    FILES ${MERGED_JSON}
    DESTINATION ${AICPU_JSON_CONFIG}
    OPTIONAL
  )
endfunction()

function(gen_aicpu_kernel_symbol enable_built_in)
  if(NOT AICPU_CUST_OBJ_TARGETS)
    message(STATUS "No aicpu cust obj targets found, skipping.")
    return()
  endif()

  set(ARM_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  set(ARM_SO_OUTPUT ${CMAKE_BINARY_DIR}/libnn_aicpu_kernels.so)

  set(ALL_OBJECTS "")
  foreach(tgt IN LISTS AICPU_CUST_OBJ_TARGETS)
    list(APPEND ALL_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  endforeach()

  message(STATUS "Linking aicpu_kernels with ARM toolchain: ${ARM_CXX_COMPILER}")
  message(STATUS "Objects: ${ALL_OBJECTS}")
  message(STATUS "Output: ${ARM_SO_OUTPUT}")

  add_custom_command(
    OUTPUT ${ARM_SO_OUTPUT}
    COMMAND ${ARM_CXX_COMPILER} -shared ${ALL_OBJECTS}
      -Wl,--whole-archive
      ${ASCEND_DIR}/lib64/libaicpu_context.a
      ${ASCEND_DIR}/lib64/libbase_ascend_protobuf.a
      -Wl,--no-whole-archive
      -Wl,-Bsymbolic
      -Wl,--exclude-libs=libbase_ascend_protobuf.a
      -Wl,-z,now
      -s
      -o ${ARM_SO_OUTPUT}
    DEPENDS ${AICPU_CUST_OBJ_TARGETS}
    COMMENT "Linking aicpu_kernels.so using ARM toolchain"
  )

  add_custom_target(aicpu_kernels ALL DEPENDS ${ARM_SO_OUTPUT})

  install(
    FILES ${ARM_SO_OUTPUT}
    DESTINATION ${AICPU_KERNEL_IMPL}
    OPTIONAL
  )
endfunction()

function(gen_onnx_plugin_symbol)
  add_library(
    ${ONNX_PLUGIN_NAME} SHARED
    $<$<TARGET_EXISTS:${ONNX_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${ONNX_PLUGIN_NAME}_obj>>
  )

  target_link_libraries(
    ${ONNX_PLUGIN_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
            c_sec
            -Wl,--no-as-needed
            register
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
    )

  install(
    TARGETS ${ONNX_PLUGIN_NAME}
    LIBRARY DESTINATION ${ONNX_PLUGIN_LIB_INSTALL_DIR}
    )

endfunction()

function(gen_norm_symbol)
  gen_ophost_symbol()

  gen_opgraph_symbol()

  gen_opapi_symbol()

  gen_onnx_plugin_symbol()
endfunction()

function(gen_cust_symbol)
  gen_cust_opapi_symbol()

  gen_cust_optiling_symbol()

  gen_cust_proto_symbol()

  gen_aicpu_json_symbol(FALSE)

  gen_aicpu_kernel_symbol(FALSE)
endfunction()
