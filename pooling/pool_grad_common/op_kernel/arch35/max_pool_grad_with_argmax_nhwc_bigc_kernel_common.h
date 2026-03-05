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
* \file max_pool_grad_with_argmax_nhwc_bigc_kernel_common.h
* \brief
*/

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_NHWC_BIGC_KERNEL_COMMON_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_NHWC_BIGC_KERNEL_COMMON_H_

#include "max_pool_grad_with_argmax_nhwc_kernel_common.h"

namespace MaxPoolGradWithArgmaxNHWCNameSpace
{
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE = 0, const uint32_t VER = VER_NORMAL>
class MaxPoolGradWithArgmaxKernelNHWCBigcBase : public MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER>
{
public:
__aicore__ inline void ConCProcVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutput = this->wOutput_;
    int64_t cOutput = this->cOutput_;

    uint16_t nOutputActual = static_cast<uint16_t>(this->nOutputActual_);
    int64_t hOutputActual = this->hOutputActual_;
    int64_t wOutputActual = this->wOutputActual_;

    int64_t curHIndex = this->hAxisIndex_ * this->hOutputInner_;
    int64_t curWIndex = this->wAxisIndex_ * this->wOutputInner_;

    uint16_t hArgmaxActual = this->hArgmaxActual_;
    uint16_t wArgmaxActual = this->wArgmaxActual_;
    uint16_t cOutputAligned = this->cOutputAligned_;
    uint16_t cOutputActual = this->cOutputActual_;

    uint16_t computeSizeT2 = this->V_REG_SIZE / sizeof(T2);
    uint16_t cRepeatimes = cOutputActual / computeSizeT2;
    uint16_t cRemain = cOutputActual - cRepeatimes * computeSizeT2;
    uint16_t cRemainLoopTimes = cRemain == 0 ? 0 : 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        for (uint16_t cIdx = 0; cIdx < cRepeatimes; ++cIdx) {
            uint32_t cOffset = cIdx * computeSizeT2;
            for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
                uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
                    for (uint16_t wIdx = 0; wIdx < wArgmaxActual; ++wIdx) {
                        uint32_t argmaxOffset =
                            ((nIdx * hArgmaxActual + hIdx) * wArgmaxActual + wIdx) * cOutputAligned + cOffset;
                        DoSingleCNhwc<T1, T2, T3, IS_CHECK_RANGE, VER>(yAddr, gradAddr, argmaxAddr, argmaxOffset,
                                                                computeSizeT2, curHIndex, curWIndex, wOutputActual,
                                                                cOutputAligned, cOffset, nOffset, cOutputActual, cOutput,
                                                                zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                    }
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        // cRemain
        for (uint16_t cIdx = 0; cIdx < cRemainLoopTimes; ++cIdx) {
            uint32_t cOffset = cRepeatimes * computeSizeT2;
            for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
                uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
                    for (uint16_t wIdx = 0; wIdx < wArgmaxActual; ++wIdx) {
                        uint32_t argmaxOffset =
                            ((nIdx * hArgmaxActual + hIdx) * wArgmaxActual + wIdx) * cOutputAligned + cOffset;
                        DoSingleCNhwc<T1, T2, T3, IS_CHECK_RANGE, VER>(yAddr, gradAddr, argmaxAddr, argmaxOffset, cRemain,
                                                                curHIndex, curWIndex, wOutputActual, cOutputAligned,
                                                                cOffset, nOffset, cOutputActual, cOutput, zeroConstReg,
                                                                wMaxReg, hMaxReg, wOutputConstReg);
                    }
                }
            }
        }
    }
}

__aicore__ inline void Compute()
{
    uint32_t calCount = this->outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = (this->outputQue_).template AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);

    LocalTensor<T1> gradLocal = (this->gradQue_).template DeQue<T1>();
    LocalTensor<T2> argmaxLocal = (this->argmaxQue_).template DeQue<T2>();

    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    ConCProcVF(yAddr, gradAddr, argmaxAddr);

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    (this->outputQue_).template EnQue(yLocal);
    (this->gradQue_).template FreeTensor(gradLocal);
    (this->argmaxQue_).template FreeTensor(argmaxLocal);
}

__aicore__ inline void ProcessPerLoop()
{
    if (this->hArgmaxActual_ <= 0 || this->wArgmaxActual_ <= 0) {
        this->ProcessNoArgmaxBlock();
        return;
    }

    this->CopyIn();
    Compute();
    this->CopyOut();
}

__aicore__ inline void Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < this->curCoreProcessNum_; loopNum++) {
        this->ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}
};
}  // namespace MaxPoolGradWithArgmaxNHWCNameSpace
#endif  // MAX_POOL_GRAD_WITH_ARGMAX_NHWC_BIGC_KERNEL_COMMON_H_