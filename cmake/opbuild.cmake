# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ######################################################################################################################
# 调用opbuild工具，生成aclnn/aclnnInner/.ini的算子信息库 等文件 generate outpath: ${ASCEND_AUTOGEN_PATH}/${sub_dir}
# ######################################################################################################################
function(gen_aclnn_classify host_obj prefix ori_out_srcs ori_out_headers opbuild_out_srcs opbuild_out_headers)
  get_target_property(module_sources ${host_obj} INTERFACE_SOURCES)
  set(sub_dir)
  # aclnn\aclnnExc以aclnn开头，aclnnInner以aclnnInner开头
  if("${prefix}" STREQUAL "aclnn")
    set(need_gen_aclnn 1)
  elseif("${prefix}" STREQUAL "aclnnInner")
    set(sub_dir inner)
    set(need_gen_aclnn 1)
  elseif("${prefix}" STREQUAL "aclnnExc")
    set(sub_dir exc)
    set(need_gen_aclnn 0)
  else()
    message(FATAL_ERROR "UnSupported aclnn prefix type, must be in aclnn/aclnnInner/aclnnExc")
  endif()

  set(out_src_path ${ASCEND_AUTOGEN_PATH}/${sub_dir})
  file(MAKE_DIRECTORY ${out_src_path})
  get_filename_component(out_src_path ${out_src_path} REALPATH)
  # opbuild_gen_aclnn/opbuild_gen_aclnnInner/opbuild_gen_aclnnExc
  if(module_sources AND ENABLE_GEN_ACLNN)
    if (module_sources)
      npu_op_code_gen(
        SRC ${module_sources}
        PACKAGE opbuild_gen_${prefix}
        OPTIONS OPS_PROTO_SEPARATE 1 OPS_PROJECT_NAME ${prefix} OPS_ACLNN_GEN ${need_gen_aclnn}
        OUT_DIR ${out_src_path}
      )
      execute_process(COMMAND bash -c "rm -fr ${out_src_path}/*.ini"
        COMMAND 
          ${CMAKE_COMMAND} -E env OPS_PROTO_SEPARATE=1 OPS_PROJECT_NAME=${prefix} OPS_ACLNN_GEN=${need_gen_aclnn}
          OPS_PRODUCT_NAME=${ASCEND_COMPUTE_UNIT} ${OP_BUILD_TOOL} ${out_src_path}/libascend_all_ops.so ${out_src_path}
      )
    endif()
  else()
    message(STATUS "No ${prefix} srcs, skip ${prefix}")
  endif()

  if("${prefix}" STREQUAL "aclnnExc")
    get_target_property(exclude_headers ${OPHOST_NAME}_aclnn_exclude_headers INTERFACE_SOURCES)
    if(exclude_headers)
      set(${opbuild_out_headers} ${ori_out_headers} ${exclude_headers} PARENT_SCOPE)
    endif()
  else()
    file(GLOB out_srcs ${out_src_path}/${prefix}_*.cpp)
    file(GLOB out_headers ${out_src_path}/${prefix}_*.h)

    set(${opbuild_out_srcs} ${ori_out_srcs} ${out_srcs} PARENT_SCOPE)
    if("${prefix}" STREQUAL "aclnn")
      set(${opbuild_out_headers} ${ori_out_headers} ${out_headers} PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(gen_aclnn_master_header aclnn_master_header_name aclnn_master_header opbuild_out_headers)
  # 规范化，防止生成的代码编译失败
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" aclnn_master_header_name "${aclnn_master_header_name}")
  string(TOUPPER ${aclnn_master_header_name} aclnn_master_header_name)

  # 生成include内容
  set(aclnn_all_header_include_content "")
  foreach(header_file ${opbuild_out_headers})
    get_filename_component(header_name ${header_file} NAME)
    set(aclnn_all_header_include_content "${aclnn_all_header_include_content}#include \"${header_name}\"\n")
  endforeach()

  # 根据模板生成头文件
  message(STATUS "create aclnn master header file: ${aclnn_master_header}")
  configure_file(
    "${OPS_NN_CMAKE_DIR}/aclnn_ops_nn.h.in"
    "${aclnn_master_header}"
    @ONLY
  )
endfunction()

function(gen_aclnn_with_opdef)
  set(opbuild_out_srcs)
  set(opbuild_out_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_obj aclnn "${opbuild_out_srcs}" "${opbuild_out_headers}"
                     opbuild_out_srcs opbuild_out_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_inner_obj aclnnInner "${opbuild_out_srcs}" "${opbuild_out_headers}"
                     opbuild_out_srcs opbuild_out_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_exclude_obj aclnnExc "${opbuild_out_srcs}" "${opbuild_out_headers}"
                     opbuild_out_srcs opbuild_out_headers)

  # 创建汇总头文件
  if(ENABLE_CUSTOM)
    set(aclnn_master_header_name "aclnn_ops_nn_${VENDOR_NAME}")
  else()
    set(aclnn_master_header_name "aclnn_ops_nn")
  endif()
  set(aclnn_master_header "${CMAKE_CURRENT_BINARY_DIR}/${aclnn_master_header_name}.h")
  gen_aclnn_master_header(${aclnn_master_header_name} "${aclnn_master_header}" "${opbuild_out_headers}")

  # 将头文件安装到packages/vendors/vendor_name/op_api/include
  if(ENABLE_PACKAGE AND NOT NO_ACLNN)
    install(FILES ${opbuild_out_headers} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    install(FILES ${opbuild_out_headers} DESTINATION ${ACLNN_OP_INC_INSTALL_DIR} OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${ACLNN_OP_INC_INSTALL_DIR} OPTIONAL)
    if(ENABLE_STATIC)
      # 将头文件安装到静态库目录
      install(FILES ${opbuild_out_headers} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include OPTIONAL)
      install(FILES ${opbuild_out_headers} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop OPTIONAL)
    endif()
  endif()

  if(opbuild_out_srcs)
    set_source_files_properties(${opbuild_out_srcs} PROPERTIES GENERATED TRUE)
    if (NOT TARGET opbuild_gen_aclnn_all)
      add_library(opbuild_gen_aclnn_all OBJECT ${opbuild_out_srcs})
    endif()
    if (ENABLE_STATIC AND NOT TARGET opbuild_custom_gen_aclnn_all)
      add_custom_target(opbuild_custom_gen_aclnn_all
                        COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/util/modify_gen_aclnn_static.py ${CMAKE_BINARY_DIR})
      add_dependencies(opbuild_gen_aclnn_all opbuild_custom_gen_aclnn_all)
    endif()
    target_include_directories(opbuild_gen_aclnn_all PRIVATE ${OPAPI_INCLUDE})
  endif()
endfunction()
