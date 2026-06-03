# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(quant_batch_matmul_inplace_add "ascend950pr_9599" "-D__CCE_AICORE__=310" "MULTI_KERNEL_TARGET")

    set(QBMIIA_UT_CMCT_BLOCK_HEADER "${CMAKE_BINARY_DIR}/tbe/ascendc/common/cmct/block/block_mmad_a8w8_fixpipe_quant.h")
    set(QBMIIA_UT_CMCT_BLOCK_SRC "${PROJECT_SOURCE_DIR}/matmul/common/cmct/block/block_mmad_a8w8_fixpipe_quant.h")
    set(QBMIIA_UT_CMCT_MX_BLOCK_HEADER "${CMAKE_BINARY_DIR}/tbe/ascendc/common/cmct/block/block_mmad_mx.h")
    set(QBMIIA_UT_CMCT_MX_BLOCK_SRC "${PROJECT_SOURCE_DIR}/matmul/common/cmct/block/block_mmad_mx.h")

    if(TARGET ascendc_kernel_ut_src_copy_quant_batch_matmul_inplace_add)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/qbmmia_cmct_block_patched.stamp"
            VERBATIM
            COMMAND ${CMAKE_COMMAND} -E copy "${QBMIIA_UT_CMCT_BLOCK_SRC}" "${QBMIIA_UT_CMCT_BLOCK_HEADER}"
            COMMAND bash -c "sed -i 's|AscendC::LocalTensor<L0CType> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};|AscendC::LocalTensor<L0CType> c1Local_{AscendC::TPosition::CO1, 0, 65536U / sizeof(L0CType)};|' \"$1\"" -- "${QBMIIA_UT_CMCT_BLOCK_HEADER}"
            COMMAND bash -c "sed -i 's|AscendC::LocalTensor<BiasType> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};|AscendC::LocalTensor<BiasType> biasBt_{AscendC::TPosition::C2, 0, 1024U};|' \"$1\"" -- "${QBMIIA_UT_CMCT_BLOCK_HEADER}"
            COMMAND ${CMAKE_COMMAND} -E copy "${QBMIIA_UT_CMCT_MX_BLOCK_SRC}" "${QBMIIA_UT_CMCT_MX_BLOCK_HEADER}"
            COMMAND bash -c "sed -i 's|AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};|AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE / sizeof(float)};|' \"$1\"" -- "${QBMIIA_UT_CMCT_MX_BLOCK_HEADER}"
            COMMAND bash -c "sed -i 's|AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};|AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, 1024U};|' \"$1\"" -- "${QBMIIA_UT_CMCT_MX_BLOCK_HEADER}"
            COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/qbmmia_cmct_block_patched.stamp"
            DEPENDS ascendc_kernel_ut_src_copy_quant_batch_matmul_inplace_add
                "${QBMIIA_UT_CMCT_BLOCK_SRC}"
                "${QBMIIA_UT_CMCT_MX_BLOCK_SRC}"
            COMMENT "Patch CMCT block header for QuantBatchMatmulInplaceAdd kernel UT"
        )
        add_custom_target(qbmmia_cmct_block_ut_patch DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/qbmmia_cmct_block_patched.stamp")
    endif()

    foreach(_qbmmia_ut_target
        opkernel_test_quant_batch_matmul_inplace_add
        opkernel_test_quant_batch_matmul_inplace_add_apt_hif8
        opkernel_test_quant_batch_matmul_inplace_add_apt_mx)
        if(TARGET ${_qbmmia_ut_target})
            if(TARGET qbmmia_cmct_block_ut_patch)
                add_dependencies(${_qbmmia_ut_target} qbmmia_cmct_block_ut_patch)
            endif()
        endif()
    endforeach()

    add_custom_target(qbmmia_ut_clean_gcda
      COMMAND ${CMAKE_COMMAND} -E rm -f
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/opkernel_test_quant_batch_matmul_inplace_add.dir/arch35/test_quant_batch_matmul_inplace_add.cpp.gcda"
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/opkernel_test_quant_batch_matmul_inplace_add_apt_hif8.dir/arch35/test_quant_batch_matmul_inplace_add_apt_hif8.cpp.gcda"
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/opkernel_test_quant_batch_matmul_inplace_add_apt_mx.dir/arch35/test_quant_batch_matmul_inplace_add_apt_mx.cpp.gcda"
    )
    foreach(_qbmmia_ut_target
        opkernel_test_quant_batch_matmul_inplace_add
        opkernel_test_quant_batch_matmul_inplace_add_apt_hif8
        opkernel_test_quant_batch_matmul_inplace_add_apt_mx)
        if(TARGET ${_qbmmia_ut_target})
            add_dependencies(${_qbmmia_ut_target} qbmmia_ut_clean_gcda)
        endif()
    endforeach()
endif()
