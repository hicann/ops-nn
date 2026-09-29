# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
    DEPENDENCIES quant_batch_matmul_v3
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_matmul_activation_quant.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend950pr_9599"
        ""
        quant_batch_matmul_v3
        MULTI_KERNEL_TARGET
    )

    set(qmmaq_kernel_ut opkernel_test_quant_matmul_activation_quant)
    if(TARGET ${qmmaq_kernel_ut})
        set(qmmaq_tensor_ut_dir "${OPTENSOR_SOURCE_PATH}/tests/ut/op_kernel")
        foreach(qmmaq_support_file blaze_kernel_stub.h kernel_ut_runner.h k3_pvwrap.cpp)
            if(NOT EXISTS "${qmmaq_tensor_ut_dir}/${qmmaq_support_file}")
                message(FATAL_ERROR "Missing ops-tensor kernel UT support: ${qmmaq_tensor_ut_dir}/${qmmaq_support_file}")
            endif()
        endforeach()
        # Prefer ops-tensor's full Blaze CPU-debug definitions over the minimal
        # NN stub. The operator stub supplies a real GM-to-L1 copy adapter.
        target_include_directories(${qmmaq_kernel_ut} BEFORE PRIVATE "${qmmaq_tensor_ut_dir}")

        # Retain AddOpTestCase's include paths, simulator flags, links and source
        # copy dependencies when compiling the same test for the other dtype/layout variants.
        foreach(qmmaq_variant E5M2 E5M2_GELU E4M3_WEIGHT_NZ)
            set(qmmaq_variant_target "${qmmaq_kernel_ut}_${qmmaq_variant}")
            add_library(${qmmaq_variant_target} OBJECT
                tests/ut/op_kernel/arch35/test_quant_matmul_activation_quant.cpp)
            foreach(qmmaq_property COMPILE_OPTIONS COMPILE_DEFINITIONS INCLUDE_DIRECTORIES LINK_LIBRARIES)
                get_target_property(qmmaq_value ${qmmaq_kernel_ut} ${qmmaq_property})
                if(qmmaq_value)
                    set_property(TARGET ${qmmaq_variant_target} PROPERTY ${qmmaq_property} "${qmmaq_value}")
                endif()
            endforeach()
            get_target_property(qmmaq_dependencies ${qmmaq_kernel_ut} MANUALLY_ADDED_DEPENDENCIES)
            if(qmmaq_dependencies)
                add_dependencies(${qmmaq_variant_target} ${qmmaq_dependencies})
            endif()
            target_compile_definitions(${qmmaq_variant_target} PRIVATE "QMMAQ_UT_${qmmaq_variant}=1")
            set_property(GLOBAL APPEND PROPERTY OPKERNEL_OBJECT_LIBS ${qmmaq_variant_target})
        endforeach()

        # The dtype/layout objects share one set of simulator PV symbols.
        target_sources(${qmmaq_kernel_ut} PRIVATE "${qmmaq_tensor_ut_dir}/k3_pvwrap.cpp")
    endif()
endif()
