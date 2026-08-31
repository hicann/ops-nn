/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool_grad_compute.h
 * \brief AdaptiveAvgPool2DGrad/AdaptiveAvgPool3DGrad big kernel 共用的梯度寄存器累加接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_AVG_POOL_GRAD_COMPUTE_H_
#define POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_AVG_POOL_GRAD_COMPUTE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace Compute {

/*
 * 功能：把当前输入点的梯度标量累加到已 gather 到寄存器的输出梯度上。
 */
template <typename COMPUTE_TYPE>
__aicore__ inline void DoGradRegAdds(AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
                                     COMPUTE_TYPE& gradInputValue, __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::Reg::MaskReg pregU32 = AscendC::Reg::UpdateMask<uint32_t>(maskCountTemp);
    AscendC::Reg::Adds(gradOutputUbValue, gradOutputUbValue, COMPUTE_TYPE(gradInputValue), pregU32);
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_AVG_POOL_GRAD_COMPUTE_H_
