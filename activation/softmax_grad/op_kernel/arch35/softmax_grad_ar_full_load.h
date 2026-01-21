/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file softmax_grad_ar_full_load.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_AR_FULL_LOAD_H
#define SOFTMAX_GRAD_AR_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "softmax_grad_base.h"

namespace SoftmaxGradOps
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

static constexpr uint32_t DOUBLE_BUFFER = 2;
static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
static constexpr uint32_t AR_FULL_LOAD_BINARY_TMP_BYTES = 512;

template <typename T>
class SoftmaxGradAR : public SoftmaxGradOpsBase
{
public:
    __aicore__ inline SoftmaxGradAR(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, const SoftmaxGradARTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessUB(int64_t ubA, int64_t aOffset);

    __aicore__ inline void NormCompute(const int64_t aSize);

    __aicore__ inline void NormComputePost(const LocalTensor<T>& dstTensor, const LocalTensor<T>& x0Tensor,
                                           const LocalTensor<T>& x1Tensor, const LocalTensor<float>& binAddTmpTensor,
                                           const int64_t aSize, const int64_t rSize, const int64_t stride);

    __aicore__ inline void NormComputeSmallR(const int64_t aSize);

    __aicore__ inline void CopyInX(int64_t ubA, int64_t offset);

    __aicore__ inline void CopyOutY(int64_t ubA, int64_t offset);

    __aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg,
                                                   uint32_t offset);

    __aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
                                                 uint32_t offset);

private:
    /* global memory address */
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> yGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> x0Queue_;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<> binaryTmpLocalBuffer_;

    int64_t blockA_ = 0;  // 获取分块操作中的单个块的大小
    const SoftmaxGradARTilingData* tl_ = nullptr;
};

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y,
                                              const SoftmaxGradARTilingData* tilingData)
{
    tl_ = tilingData;

    // 获取分块操作中的单个块的大小。判断是否是最后一块，是最后一块，则等于剩余元素的数量，否则等于固定的单核处理的行数
    blockA_ = (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1)
                  ? (tl_->a - tl_->aBlockFactor * (AscendC::GetBlockNum() - 1))
                  : tl_->aBlockFactor;

    // 初始化GM Tensor
    int64_t aGmOffset = tl_->aBlockFactor * AscendC::GetBlockIdx() * tl_->r;
    x0Gm_.SetGlobalBuffer((__gm__ T*)x0 + aGmOffset);
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1 + aGmOffset);
    yGm_.SetGlobalBuffer((__gm__ T*)y + aGmOffset);

    // 初始化Pipe
    int64_t ubBufferSize = tl_->ubFactor * tl_->rAligned;
    pipe_->InitBuffer(x0Queue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(x1Queue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(binaryTmpLocalBuffer_, AR_FULL_LOAD_BINARY_TMP_BYTES);
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::Process()
{
    // ubLoop: 表示需要多少个子块来覆盖singleA大小的数据
    int64_t ubLoop = ops::CeilDiv(blockA_, tl_->ubFactor);
    int64_t lastUbFactor = blockA_ - tl_->ubFactor * (ubLoop - 1);
    // 循环处理每个子块
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoop; ubLoopIdx++) {
        // aOffset：计算当前子块的偏移量
        int64_t aOffset = ubLoopIdx * tl_->ubFactor * tl_->r;
        int64_t ubA = (ubLoopIdx == (ubLoop - 1)) ? lastUbFactor : tl_->ubFactor;
        ProcessUB(ubA, aOffset);
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::ProcessUB(int64_t ubA, int64_t aOffset)
{
    if (ubA <= 0 || tl_->r <= 0) {
        return;
    }

    CopyInX(ubA, aOffset);

    if (tl_->r <= CONST_TWO * VL_FP32) {
        NormComputeSmallR(ubA);
    } else {
        NormCompute(ubA);
    }

    CopyOutY(ubA, aOffset);
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::NormComputeSmallR(const int64_t aSize)
{
    uint32_t rSize = tl_->r;
    uint32_t rAligned = tl_->rAligned;

    LocalTensor<T> x0Tensor = x0Queue_.DeQue<T>();
    LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();
    LocalTensor<T> dstTensor = yQueue_.AllocTensor<T>();

    uint16_t loopTimes = aSize;
    if (rSize <= VL_FP32) {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* x0 = (__local_mem__ T*)x0Tensor.GetPhyAddr();
        __local_mem__ T* x1 = (__local_mem__ T*)x1Tensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; i++) {
                LoadTensorForDtypeTIn(x0, reg0, pMask, i * rAligned);
                LoadTensorForDtypeTIn(x1, reg1, pMask, i * rAligned);
                Mul(reg2, reg0, reg1, pMask);

                ReduceSum(reg2, reg2, pMask);
                Duplicate(reg2, reg2, pFull);

                Mul(reg1, reg0, reg1, pMask);
                Mul(reg0, reg0, reg2, pMask);
                Sub(reg1, reg1, reg0, pMask);

                StoreTensorForDtypeTOut(dst, reg1, pMask, i * rAligned);
            }
        }
    } else {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* x0 = (__local_mem__ T*)x0Tensor.GetPhyAddr();
        __local_mem__ T* x1 = (__local_mem__ T*)x1Tensor.GetPhyAddr();
        __local_mem__ T* x0_1 = (__local_mem__ T*)x0Tensor.GetPhyAddr() + VL_FP32;
        __local_mem__ T* x1_1 = (__local_mem__ T*)x1Tensor.GetPhyAddr() + VL_FP32;

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; i++) {
                LoadTensorForDtypeTIn(x0, reg0, pFull, i * rAligned);
                LoadTensorForDtypeTIn(x1, reg1, pFull, i * rAligned);
                LoadTensorForDtypeTIn(x0_1, reg0_1, pMask, i * rAligned);
                LoadTensorForDtypeTIn(x1_1, reg1_1, pMask, i * rAligned);

                Mul(reg2, reg0, reg1, pFull);
                Mul(reg2_1, reg0_1, reg1_1, pMask);

                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                ReduceSum(reg2, reg2, pFull);
                Duplicate(reg2, reg2, pFull);

                Mul(reg1, reg0, reg1, pFull);
                Mul(reg0, reg0, reg2, pFull);
                Sub(reg1, reg1, reg0, pFull);
                StoreTensorForDtypeTOut(dst, reg1, pFull, i * rAligned);

                Mul(reg1_1, reg0_1, reg1_1, pMask);
                Mul(reg0_1, reg0_1, reg2, pMask);
                Sub(reg1_1, reg1_1, reg0_1, pMask);
                StoreTensorForDtypeTOut(dst, reg1_1, pMask, i * rAligned + VL_FP32);
            }
        }
    }

    yQueue_.EnQue(dstTensor);
    x0Queue_.FreeTensor<T>(x0Tensor);
    x1Queue_.FreeTensor<T>(x1Tensor);
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::NormCompute(const int64_t aSize)
{
    LocalTensor<T> x0Tensor = x0Queue_.DeQue<T>();
    LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();
    LocalTensor<T> dstTensor = yQueue_.AllocTensor<T>();
    LocalTensor<float> reduceSumTempTensor = binaryTmpLocalBuffer_.AllocTensor<float>();

    int64_t ceilVLCount =
        ops::CeilDiv(static_cast<int64_t>(tl_->r * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t floorVLCount =
        ops::FloorDiv(static_cast<int64_t>(tl_->r * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(tl_->r - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = tl_->rAligned;
    uint32_t innerLoopStride = VL_FP32;
    uint32_t outerLoopDstStride =
        ops::Aligned(static_cast<int64_t>(foldPoint), static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __local_mem__ float* dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr();
    __local_mem__ T* foldX0A = (__local_mem__ T*)x0Tensor.GetPhyAddr();
    __local_mem__ T* foldX0B = (__local_mem__ T*)x0Tensor.GetPhyAddr() + foldSrcBOffset;
    __local_mem__ T* tailX0A = (__local_mem__ T*)x0Tensor.GetPhyAddr() + tailSrcAOffset;
    __local_mem__ T* tailX0B = (__local_mem__ T*)x0Tensor.GetPhyAddr() + tailSrcBOffset;
    __local_mem__ T* unFoldX0 = (__local_mem__ T*)x0Tensor.GetPhyAddr() + unFoldSrcOffset;

    __local_mem__ T* foldX1A = (__local_mem__ T*)x1Tensor.GetPhyAddr();
    __local_mem__ T* foldX1B = (__local_mem__ T*)x1Tensor.GetPhyAddr() + foldSrcBOffset;
    __local_mem__ T* tailX1A = (__local_mem__ T*)x1Tensor.GetPhyAddr() + tailSrcAOffset;
    __local_mem__ T* tailX1B = (__local_mem__ T*)x1Tensor.GetPhyAddr() + tailSrcBOffset;
    __local_mem__ T* unFoldX1 = (__local_mem__ T*)x1Tensor.GetPhyAddr() + unFoldSrcOffset;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg UReg;

        for (uint16_t i = 0; i < outerLoopTimes; i++) {
            dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; j++) {
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
                LoadTensorForDtypeTIn(foldX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(foldX0B, reg1, pFull, i * outerLoopStride + j * innerLoopStride);

                LoadTensorForDtypeTIn(foldX1A, reg0_1, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(foldX1B, reg1_1, pFull, i * outerLoopStride + j * innerLoopStride);

                Mul(reg2, reg0, reg0_1, pFull);
                Mul(reg2_1, reg1, reg1_1, pFull);

                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2, reg2, reg2_1, pFull);
                ReduceSum(reg2, reg2, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, reg2, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; j++) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg0_1, reg1_1, reg2_1;
                AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);

                LoadTensorForDtypeTIn(tailX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(tailX0B, reg1, pMask, i * outerLoopStride + j * innerLoopStride);

                LoadTensorForDtypeTIn(tailX1A, reg0_1, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(tailX1B, reg1_1, pMask, i * outerLoopStride + j * innerLoopStride);

                Mul(reg2, reg0, reg0_1, pFull);
                Mul(reg2_1, reg1, reg1_1, pMask);

                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                ReduceSum(reg2, reg2, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, reg2, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; j++) {
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg0_1;
                LoadTensorForDtypeTIn(unFoldX0, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(unFoldX1, reg0_1, pFull, i * outerLoopStride + j * innerLoopStride);

                Mul(reg1, reg0, reg0_1, pFull);
                ReduceSum(reg1, reg1, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, reg1, UReg, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    }
    NormComputePost(dstTensor, x0Tensor, x1Tensor, reduceSumTempTensor, aSize, foldPoint, outerLoopDstStride);

    yQueue_.EnQue(dstTensor);
    x0Queue_.FreeTensor<T>(x0Tensor);
    x1Queue_.FreeTensor<T>(x1Tensor);
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::NormComputePost(const LocalTensor<T>& dstTensor,
                                                         const LocalTensor<T>& x0Tensor, const LocalTensor<T>& x1Tensor,
                                                         const LocalTensor<float>& binAddTmpTensor, const int64_t aSize,
                                                         const int64_t rSize, const int64_t stride)
{
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = tl_->rLoopCount;
    uint16_t oriR = tl_->r;
    uint16_t oriRAligned = tl_->rAligned;

    if (rSize <= VL_FP32) {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* x0 = (__local_mem__ T*)x0Tensor.GetPhyAddr();
        __local_mem__ T* x1 = (__local_mem__ T*)x1Tensor.GetPhyAddr();
        __local_mem__ float* sumTmp = (__local_mem__ float*)binAddTmpTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; i++) {
                DataCopy(reg0, (__local_mem__ float*)sumTmp + i * static_cast<uint32_t>(stride));
                ReduceSum(reg1, reg0, pMask);
                Duplicate(reg2, reg1, pFull);

                uint32_t sreg0 = static_cast<uint32_t>(oriR);
                for (uint16_t j = 0; j < rLoopCount; j++) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t offset = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeTIn(x0, reg0, maskOri, offset);
                    LoadTensorForDtypeTIn(x1, reg1, maskOri, offset);
                    Mul(reg1, reg0, reg1, maskOri);
                    Mul(reg0, reg0, reg2, maskOri);
                    Sub(reg1, reg1, reg0, maskOri);
                    StoreTensorForDtypeTOut(dst, reg1, maskOri, offset);
                }
            }
        }
    } else {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ float* sumTmpA = (__local_mem__ float*)binAddTmpTensor.GetPhyAddr();
        __local_mem__ float* sumTmpB = (__local_mem__ float*)binAddTmpTensor.GetPhyAddr() + VL_FP32;

        __local_mem__ T* x0 = (__local_mem__ T*)x0Tensor.GetPhyAddr();
        __local_mem__ T* x1 = (__local_mem__ T*)x1Tensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; i++) {
                DataCopy(reg0, (__local_mem__ float*)sumTmpA + i * static_cast<uint32_t>(stride));
                DataCopy(reg1, (__local_mem__ float*)sumTmpB + i * static_cast<uint32_t>(stride));
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                ReduceSum(reg2, reg0, pFull);
                Duplicate(reg2, reg2, pFull);
                uint32_t sreg0 = static_cast<uint32_t>(oriR);
                for (uint16_t j = 0; j < rLoopCount; j++) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t offset = j * VL_FP32 + i * oriRAligned;

                    LoadTensorForDtypeTIn(x0, reg0, maskOri, offset);
                    LoadTensorForDtypeTIn(x1, reg1, maskOri, offset);

                    Mul(reg1, reg0, reg1, maskOri);
                    Mul(reg0, reg0, reg2, maskOri);
                    Sub(reg1, reg1, reg0, maskOri);
                    StoreTensorForDtypeTOut(dst, reg1, maskOri, offset);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::LoadTensorForDtypeTIn(__local_mem__ T* src, RegTensor<float>& dst,
                                                               MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else {  // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::CopyInX(int64_t ubA, int64_t offset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams copyInParams;
    copyInParams.blockCount = ubA;
    copyInParams.blockLen = tl_->r * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (tl_->rAligned - tl_->r) * sizeof(T) / BLOCK_SIZE;

    LocalTensor<T> x0InUb = x0Queue_.AllocTensor<T>();
    DataCopyPad(x0InUb, x0Gm_[offset], copyInParams, padParams);
    x0Queue_.EnQue(x0InUb);

    LocalTensor<T> x1InUb = x1Queue_.AllocTensor<T>();
    DataCopyPad(x1InUb, x1Gm_[offset], copyInParams, padParams);
    x1Queue_.EnQue(x1InUb);
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::StoreTensorForDtypeTOut(__local_mem__ T* dst,
                                                                 AscendC::MicroAPI::RegTensor<float>& src,
                                                                 AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitFp32ToFp16>(xFp16, src, preg);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradAR<T>::CopyOutY(int64_t ubA, int64_t offset)
{
    DataCopyParams copyOutParams;
    copyOutParams.blockCount = ubA;
    copyOutParams.blockLen = tl_->r * sizeof(T);
    copyOutParams.srcStride = (tl_->rAligned - tl_->r) * sizeof(T) / BLOCK_SIZE;
    copyOutParams.dstStride = 0;

    LocalTensor<T> yOutUb = yQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yOutUb, copyOutParams);
    yQueue_.FreeTensor(yOutUb);
}

}  // namespace SoftmaxGradOps
#endif  // SOFTMAX_GRAD_AR_FULL_LOAD_H
