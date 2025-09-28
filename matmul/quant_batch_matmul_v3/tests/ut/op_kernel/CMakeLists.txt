 # -----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 # -----------------------------------------------------------------------------
return()
if(UT_TEST_ALL OR FAST_OP_UT)

include_directories(
        ${ASCEND_DIR}/opensdk/opensdk/gtest/include
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
        ${CANN_ROOT}/ops/built-in/tbe/impl/ascendc/common/
        ${CANN_ROOT}/ops/built-in/tests/ut/op_tiling_test
        ${ASCEND_DIR}/runtime/highlevel_api
        ${ASCEND_DIR}/include/ascendc/highlevel_api
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${CANN_ROOT}/ops/common
        ${CANN_ROOT}/ops/built-in/tests/ut/fast_op_test
        )

SET(PRIVATE_L
    c_sec
    mmpa
    error_manager
    pem_davinci
    tikicpulib::ascend910B1
    pthread
)

function(GetQmmv3OrigDtype dtype)
    if(dtype STREQUAL "uint64_t")
        set(origDtype "DT_UINT64" PARENT_SCOPE)
    elseif(dtype STREQUAL "float")
        set(origDtype "DT_FLOAT" PARENT_SCOPE)
    elseif(dtype STREQUAL "half")
        set(origDtype "DT_FLOAT16" PARENT_SCOPE)
    elseif(dtype STREQUAL "int32_t")
        set(origDtype "DT_INT32" PARENT_SCOPE)
    elseif(dtype STREQUAL "int4b_t")
        set(origDtype "DT_INT4" PARENT_SCOPE)
    elseif(dtype STREQUAL "bfloat16_t")
        set(origDtype "DT_BF16" PARENT_SCOPE)
    elseif(dtype STREQUAL "int8_t")
        set(origDtype "DT_INT8" PARENT_SCOPE)
    elseif(dtype STREQUAL "float8_e8m0_t")
        set(origDtype "DT_FLOAT8_E8M0" PARENT_SCOPE)
    elseif(dtype STREQUAL "float8_e4m3_t")
        set(origDtype "DT_FLOAT8_E4M3" PARENT_SCOPE)
    elseif(dtype STREQUAL "float8_e5m2_t")
        set(origDtype "DT_FLOAT8_E5M2" PARENT_SCOPE)
    elseif(dtype STREQUAL "float4_e1m2_t")
        set(origDtype "DT_FLOAT4_E1M2" PARENT_SCOPE)
    elseif(dtype STREQUAL "float4_e2m1_t")
        set(origDtype "DT_FLOAT4_E2M1" PARENT_SCOPE)
    endif()
endfunction()

function(GetQmmv3FmtEnum fmt)
    if(fmt STREQUAL "ND")
        set(fmtEnum "2" PARENT_SCOPE)
    else()
        set(fmtEnum "29" PARENT_SCOPE)
    endif()
endfunction()

function(AddQmmv3Case caseName socVersion x1Dtype x2Dtype scaleDtype biasDtype pertokenDtype yDtype x1Format x2Format)
    set(utTarget ${caseName}_${socVersion})
    if(socVersion STREQUAL "ascend310P1")
        set(socMacro -D__CCE_AICORE__=200)
        set(cpulibName tikicpulib)
    elseif(socVersion STREQUAL "ascend910B1")
        set(socMacro -D__CCE_AICORE__=220 -D__DAV_C220__)
        set(cpulibName tikicpulib::${socVersion})
    endif()

    GetQmmv3OrigDtype(${x1Dtype})
    set(origDtypeX1 ${origDtype})
    GetQmmv3OrigDtype(${x2Dtype})
    set(origDtypeX2 ${origDtype})
    GetQmmv3OrigDtype(${scaleDtype})
    set(origDtypeScale ${origDtype})
    GetQmmv3OrigDtype(${biasDtype})
    set(origDtypeBias ${origDtype})
    GetQmmv3OrigDtype(${pertokenDtype})
    set(origDtypePertoken ${origDtype})
    GetQmmv3OrigDtype(${yDtype})
    set(origDtypeY ${origDtype})
    GetQmmv3FmtEnum(${x1Format})
    set(fmtEnumX1 ${fmtEnum})
    GetQmmv3FmtEnum(${x2Format})
    set(fmtEnumX2 ${fmtEnum})

    add_custom_command(
        OUTPUT test_${utTarget}.cpp
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test_quant_batch_matmul_v3.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/test_${utTarget}.cpp
        COMMAND sed -i "s/QMMV3_SUITE_NAME/${caseName}/g" ${CMAKE_CURRENT_BINARY_DIR}/test_${utTarget}.cpp
    )
    add_library(${utTarget} SHARED
            ${CANN_ROOT}/ops/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/test_${utTarget}.cpp)

target_compile_options(${utTarget} PRIVATE
                            -DDTYPE_X1=${x1Dtype}
                            -DDTYPE_X2=${x2Dtype}
                            -DORIG_DTYPE_X1=${origDtypeX1}
                            -DORIG_DTYPE_X2=${origDtypeX2}
                            -DFORMAT_X1=${fmtEnumX1}
                            -DFORMAT_X2=${fmtEnumX2}
                            -DDTYPE_Y=${yDtype}
                            -DORIG_DTYPE_Y=${origDtypeY}
                            -DDTYPE_SCALE=${scaleDtype}
                            -DORIG_DTYPE_SCALE=${origDtypeScale}
                            -DDTYPE_BIAS=${biasDtype}
                            -DORIG_DTYPE_BIAS=${origDtypeBias}
                            -DDTYPE_PERTOKEN_SCALE=${pertokenDtype}
                            -DORIG_DTYPE_PERTOKEN_SCALE=${origDtypePertoken}
                            ${socMacro}
                            -g
                            -fvisibility=hidden
                            -include ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3_tiling_def.h)

target_link_libraries(${utTarget} PRIVATE
                          $<BUILD_INTERFACE:intf_llt_pub_asan>
                          -Wl,--whole-archive
                          -Wl,--no-as-needed
                          ${PRIVATE_L}
                          -Wl,--as-needed
                          -Wl,--no-whole-archive
    )

endfunction()
#AddQmmv3Case(quant_batch_matmul_v3_310p ascend910B1 int8_t int8_t uint64_t int32_t float half ND ND)
AddQmmv3Case(quant_batch_matmul_v3 ascend910B1 int8_t int8_t uint64_t int32_t float half ND ND)
AddQmmv3Case(quant_batch_matmul_v3_bf16 ascend910B1 int8_t int8_t bfloat16_t bfloat16_t float bfloat16_t ND ND)
AddQmmv3Case(quant_batch_matmul_v3_bf16_bias_fp32 ascend910B1 int8_t int8_t bfloat16_t float float bfloat16_t ND ND)
AddQmmv3Case(quant_batch_matmul_v3_bf16_scale_fp32 ascend910B1 int8_t int8_t float bfloat16_t float bfloat16_t ND ND)
AddQmmv3Case(quant_batch_matmul_v3_fp16_pertoken ascend910B1 int8_t int8_t float int32_t float half ND ND)
AddQmmv3Case(quant_batch_matmul_v3_fp16_pertoken_bias_fp32 ascend910B1 int8_t int8_t float float float half ND ND)
AddQmmv3Case(quant_batch_matmul_v3_int32 ascend910B1 int8_t int8_t bfloat16_t int32_t float int32_t ND ND)
AddQmmv3Case(quant_batch_matmul_v3_int4 ascend910B1 int4b_t int4b_t uint64_t int32_t float half ND NZ)
AddQmmv3Case(quant_batch_matmul_v3_nz ascend910B1 int8_t int8_t uint64_t int32_t float half ND NZ)
AddQmmv3Case(quant_batch_matmul_v3_x1_nz ascend910B1 int8_t int8_t float bfloat16_t float half NZ NZ)


endif()