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
    set(log_softmax_grad_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/log_softmax_grad_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/log_softmax_grad_infershape.cpp
        )
    # -fpermissive: under the CPU-emulation (ASCENDC_CPU_DEBUG) UT toolchain, LocalTensor::GetPhyAddr
    # returns a host pointer (float*) instead of the device-only uint64_t address assumed by the kernel's
    # trans-data code. The pointer bits fit losslessly into uint64_t on x86-64, so runtime stays correct.
    # -Dfloat16_t=uint16_t: the CPU-emulation toolchain's kernel_operator.h stub does not declare float16_t;
    # it is only referenced inside discarded `if constexpr` branches for T=float, so mapping it to uint16_t
    # is sufficient for compilation. (bfloat16_t is already provided by the stub.)
    AddOpTestCase(log_softmax_grad "ascend910B1" "-DDTYPE_INPUT_DY=float -fpermissive -Dfloat16_t=uint16_t" "${log_softmax_grad_tiling_files}")
endif()
