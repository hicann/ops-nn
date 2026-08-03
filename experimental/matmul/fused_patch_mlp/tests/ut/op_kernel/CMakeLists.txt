# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    set(fused_patch_mlp_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_patch_mlp_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_patch_mlp_infershape.cpp
        )
    AddOpTestCase(fused_patch_mlp "ascend910B1" "" ${fused_patch_mlp_tiling_files})

    # This UT supplies an AIC-only wrapper that instantiates KernelFusedPatchMlp from its header. Do not compile the
    # copied production entry into the same CPU-simulator target: its MIX metadata would make TmSim launch AIV
    # processes for the wrapper and execute Matmul without AIV-side L1 storage.
    set(fused_patch_mlp_copied_kernel
        ${ASCEND_KERNEL_SRC_DST}/fused_patch_mlp/fused_patch_mlp.cpp)
    set_source_files_properties(${fused_patch_mlp_copied_kernel} PROPERTIES HEADER_FILE_ONLY TRUE)
endif()
