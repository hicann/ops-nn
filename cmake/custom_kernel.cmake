# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### custom to kernel build #####

function(add_custom_kernel_library ascendc_kernels)
    unset(op_def_list)
    foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
        foreach(OP_DIR ${COMPILED_OP_DIRS})
            get_filename_component(op_name ${OP_DIR} NAME)
            set(binary_json ${OP_DIR}/op_host/config/${compute_unit}/${op_name}_binary.json)
            if(EXISTS ${binary_json})
                list(FIND ASCEND_OP_NAME ${op_name} INDEX)
                if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
                   continue()
                endif()
                get_op_type_from_binary_json(${binary_json} op_type)
                if(EXISTS ${OP_DIR}/op_kernel/${op_name}.cpp)
                    list(APPEND op_def_list "${OP_DIR}/op_host/${op_name}_def.cpp")
                    list(APPEND op_name_list "${op_name}")
                    list(APPEND op_type_list "${op_type}")
                endif()
            endif()
        endforeach()
    endforeach()

    list(LENGTH op_name_list op_name_list_LENGTH)
    if(${op_name_list_LENGTH} EQUAL 0)
        message(STATUS "No op found in ${COMPILED_OP_DIRS}")
        return()
    endif()
    math(EXPR last_index "${op_name_list_LENGTH}-1")
    if(op_def_list)
        npu_op_code_gen(
            SRC ${op_def_list}
            PACKAGE ${PACK_CUSTOM_NAME}
            COMPILE_OPTIONS -g
            OUT_DIR ${ASCEND_AUTOGEN_PATH}
            OPTIONS OPS_PRODUCT_NAME "aclnnExc"
                    OPS_PROJECT_NAME ${ASCEND_COMPUTE_UNIT}
                    OPS_ACLNN_GEN 0
        )
        npu_op_kernel_library(ascendc_kernels_lib
            SRC_BASE ${ASCEND_KERNEL_SRC_DST}
            TILING_LIBRARY cust_opmaster)

        npu_op_kernel_options(ascendc_kernels_lib ALL OPTIONS --save-temp-files)
        foreach(i RANGE ${last_index})
            list(GET op_name_list ${i} op_name_tmp)
            list(GET op_type_list ${i} op_type_tmp)
            foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
                map_compute_unit(${compute_unit} compute_unit_long)
                npu_op_kernel_sources(ascendc_kernels_lib
                    OP_TYPE ${op_type_tmp}
                    KERNEL_DIR ./${op_name_tmp}
                    COMPUTE_UNIT ${compute_unit_long}
                    KERNEL_FILE ${op_name_tmp}.cpp
                )
            endforeach()
        endforeach()
        add_dependencies(ascendc_kernels_lib_copy_kernel_srcs ascendc_kernel_src_copy)
    endif()
    set(${ascendc_kernels} ascendc_kernels_lib PARENT_SCOPE)
endfunction()