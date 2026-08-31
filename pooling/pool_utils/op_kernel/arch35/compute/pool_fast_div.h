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
 * \file pool_fast_div.h
 * \brief AvgPool/Pool3D/PoolGrad 共用的 magic number 快速无符号除法接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_POOL_FAST_DIV_H_
#define POOL_UTILS_ARCH35_COMPUTE_POOL_FAST_DIV_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace Compute {

__aicore__ inline void FastDivImpl(AscendC::Reg::RegTensor<uint32_t>& res, AscendC::Reg::RegTensor<uint32_t>& src,
                                   AscendC::Reg::RegTensor<uint32_t>& magic, int16_t shift, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<uint32_t> tmp;
    AscendC::Reg::Mull(tmp, res, src, magic, mask);
    AscendC::Reg::Add(tmp, src, res, mask);
    AscendC::Reg::ShiftRights(res, tmp, shift, mask);
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_POOL_FAST_DIV_H_
