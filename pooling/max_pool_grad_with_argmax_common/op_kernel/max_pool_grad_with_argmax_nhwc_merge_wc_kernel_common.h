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
 * \file max_pool_grad_with_argmax_nhwc_merge_wc_kernel_common.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_NHWC_MERGE_WC_KERNEL_COMMON_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_NHWC_MERGE_WC_KERNEL_COMMON_H_

#include "max_pool_grad_with_argmax_nhwc_kernel_common.h"

namespace MaxPoolGradWithArgmaxNHWCNameSpace {
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE = 0, const uint32_t VER = VER_NORMAL>
class MaxPoolGradWithArgmaxKernelNHWCMergeWCBase
    : public MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER> {
public:
    __aicore__ inline void ConCMergeWProcVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr)
    {
        int64_t wOutput = this->wOutput_;
        uint16_t cOutputActual = this->cOutputActual_;
        uint16_t cOutputAligned = this->cOutputAligned_;

        uint16_t nOutputActual = static_cast<uint16_t>(this->nOutputActual_);
        int64_t wOutputActual = this->wOutputActual_;
        int64_t hOutputActual = this->hOutputActual_;

        int64_t curHIndex = this->hAxisIndex_ * this->hOutputInner_;
        int64_t curWIndex = this->wAxisIndex_ * this->wOutputInner_;
        int64_t wArgmaxActual = this->wArgmaxActual_;
        uint16_t hArgmaxActual = this->hArgmaxActual_;

        uint16_t hProBatchSize = this->curHProBatchSize_;
        uint16_t wProBatchSize = this->curWProBatchSize_;
        uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;

        uint16_t computeSizeT2 = this->V_REG_SIZE / sizeof(T2);
        uint16_t concurrencyCount = computeSizeT2 / cOutputActual;

        uint16_t repeatimes = wFullBatchCount / concurrencyCount;
        uint16_t wRemain = wArgmaxActual - repeatimes * wProBatchSize * concurrencyCount;
        uint32_t wRemainBatch = wRemain / wProBatchSize;
        uint16_t wRemainTail = wRemain % wProBatchSize;

        uint32_t mask0 = concurrencyCount * cOutputActual;
        uint32_t mask1 = wRemainBatch * cOutputActual;
        uint32_t mask2 = 1 * cOutputActual;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
            GenInitial3DIndices(
                (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize, wArgmaxActual,
                wFullBatchCount, cOutputActual, cOutputAligned);
            AscendC::MicroAPI::MaskReg allMask =
                AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMask);
        }

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
            uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cOutputAligned;

            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                    for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset =
                                (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize + hIdx * wArgmaxActual) *
                                    cOutputAligned +
                                nArgmaxOffset;

                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE, VER>(
                                yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0, curHIndex, curWIndex,
                                wOutputActual, hOutputActual, cOutputAligned, 0, nOffset, cOutputActual,
                                wOutputConstReg);
                        }
                    }
                }
            }
        }

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
            uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cOutputAligned;

            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

                    AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                    // batch  mask
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize + hIdx * wArgmaxActual) *
                                        cOutputAligned +
                                    nArgmaxOffset;

                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE, VER>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1, curHIndex, curWIndex, wOutputActual,
                            hOutputActual, cOutputAligned, 0, nOffset, cOutputActual, wOutputConstReg);
                    }
                }
            }
        }

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
            AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

            AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
            AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

            AscendC::MicroAPI::MaskReg allMaskU32 =
                AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);

            for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
                uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
                uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cOutputAligned;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wRemainBatch * wProBatchSize +
                                     repeatimes * concurrencyCount * wProBatchSize + hIdx * wArgmaxActual) *
                                        cOutputAligned +
                                    nArgmaxOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE, VER>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2, curHIndex, curWIndex, wOutputActual,
                            hOutputActual, cOutputAligned, 0, nOffset, cOutputActual, wOutputConstReg);
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

        LocalTensor<uint32_t> helpTensor = (this->helpBuf_).template Get<uint32_t>();
        __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();

        ConCMergeWProcVF(yAddr, gradAddr, argmaxAddr, helpAddr);

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
            this->ProcessNoArgmaxBlock(); // ceilModefalse
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
} // namespace MaxPoolGradWithArgmaxNHWCNameSpace
#endif // MAX_POOL_GRAD_WITH_ARGMAX__NHWC_KERNEL_H_
