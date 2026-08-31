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
 * \file max_pool_with_argmax_nhwc_big_c_compute.h
 * \brief MaxPoolWithArgmax NHWC kernel 共用的 pad 负无穷填充接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_MAX_POOL_WITH_ARGMAX_NHWC_BIG_C_COMPUTE_H_
#define POOL_UTILS_ARCH35_COMPUTE_MAX_POOL_WITH_ARGMAX_NHWC_BIG_C_COMPUTE_H_

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "pool_utils/arch35/compute/max_pool_negative_value.h"

namespace PoolUtils {
namespace Compute {

template <typename T1>
__aicore__ inline void FillPadNegVF(__ubuf__ T1* xLocalAddr, int64_t baseBlockTopOffsetInOcean_,
                                    int64_t baseBlockLeftOffsetInOcean_, int64_t baseBlockRightOffsetInOcean_,
                                    int64_t baseBlockDownOffsetInOcean_, int64_t wInputActual_, int64_t hInputActual_,
                                    int64_t cOutputActualAlign_, int64_t hOutputActual_, int64_t hStride_,
                                    int64_t hKernel_, int64_t wOutputActual_, int64_t wStride_, int64_t wKernel_,
                                    int64_t nOutputActual_)
{
    int32_t top = baseBlockTopOffsetInOcean_;
    int32_t left = baseBlockLeftOffsetInOcean_;
    int32_t right = baseBlockRightOffsetInOcean_;
    int32_t down = baseBlockDownOffsetInOcean_;
    int64_t wInputActual = wInputActual_;
    int64_t hInputActual = hInputActual_;
    int32_t cOutputActualAlign = cOutputActualAlign_;
    int32_t hInputActualAmend = (hOutputActual_ - 1) * hStride_ + hKernel_;
    int32_t wInputActualAmend = (wOutputActual_ - 1) * wStride_ + wKernel_;
    uint32_t computeSize = Ops::Base::GetVRegSize() / sizeof(T1);

    uint32_t topCount = top * wInputActualAmend * cOutputActualAlign;
    uint16_t topRepeatTimes = (topCount + computeSize - 1) / computeSize;

    int32_t leftSingleRowCount = left * cOutputActualAlign;
    uint16_t leftSingleRowRepeatTimes = (leftSingleRowCount + computeSize - 1) / computeSize;
    int32_t leftStartOffset = topCount;

    int32_t rightSingleRowCount = right * cOutputActualAlign;
    uint16_t rightSingleRowRepeatTimes = (rightSingleRowCount + computeSize - 1) / computeSize;
    int32_t rightStartOffset = topCount + (wInputActual + left) * cOutputActualAlign;

    uint32_t downCount = down * wInputActualAmend * cOutputActualAlign;
    uint16_t downRepeatTimes = (downCount + computeSize - 1) / computeSize;
    int32_t downStartOffset = (hInputActual + top) * wInputActualAmend * cOutputActualAlign;
    uint16_t nOutputActual = nOutputActual_;
    int32_t nStartOffset = hInputActualAmend * wInputActualAmend * cOutputActualAlign;
    __VEC_SCOPE__

    {
        AscendC::Reg::RegTensor<T1> negInfReg;
        PoolUtils::Compute::DuplicateNegInfReg<T1>(negInfReg);
        for (uint16_t n = 0; n < nOutputActual; n++) {
            int32_t nOffset = n * nStartOffset;
            // top
            uint32_t topCountTmp = topCount;
            for (uint16_t i = 0; i < topRepeatTimes; i++) {
                AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<T1>(topCountTmp);
                AscendC::Reg::StoreAlign(xLocalAddr + nOffset + i * computeSize, negInfReg, preg);
            }

            // left
            for (uint16_t hIndex = 0; hIndex < static_cast<uint16_t>(hInputActual); hIndex++) {
                int32_t leftOffset = hIndex * wInputActualAmend * cOutputActualAlign + leftStartOffset;
                uint32_t leftCount = leftSingleRowCount;
                for (uint16_t i = 0; i < leftSingleRowRepeatTimes; i++) {
                    AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<T1>(leftCount);
                    AscendC::Reg::StoreAlign(xLocalAddr + nOffset + leftOffset + i * computeSize, negInfReg, preg);
                }
            }

            // right
            for (uint16_t hIndex = 0; hIndex < static_cast<uint16_t>(hInputActual); hIndex++) {
                int32_t rightOffset = hIndex * wInputActualAmend * cOutputActualAlign + rightStartOffset;
                uint32_t rightCount = rightSingleRowCount;
                for (uint16_t i = 0; i < rightSingleRowRepeatTimes; i++) {
                    AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<T1>(rightCount);
                    AscendC::Reg::StoreAlign(xLocalAddr + nOffset + rightOffset + i * computeSize, negInfReg, preg);
                }
            }

            // down
            uint32_t downCountTmp = downCount;
            for (uint16_t i = 0; i < downRepeatTimes; i++) {
                AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<T1>(downCountTmp);
                AscendC::Reg::StoreAlign(xLocalAddr + nOffset + downStartOffset + i * computeSize, negInfReg, preg);
            }
        }
    }
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_MAX_POOL_WITH_ARGMAX_NHWC_BIG_C_COMPUTE_H_
