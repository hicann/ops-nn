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
 * \file instance_norm_ar_welford.h
 * \brief
 */


#ifndef INSTANCE_NORM_AR_WELFORD_H_
#define INSTANCE_NORM_AR_WELFORD_H_

#include "instance_norm_common.h"
#include "op_kernel/platform_util.h"

namespace InstanceNormOps {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, const int64_t rowSize)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const int64_t rowSize)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T, typename T_BETA, typename T_MEAN>
class InstanceNormARWelford {
public:
    __aicore__ inline uint64_t CEIL_DIV(uint64_t x, uint64_t y)
    {
        return (y != 0) ? (x + y - 1) / y : 0;
    }
    
    __aicore__ inline InstanceNormARWelford(const InstanceNormARWelfordTilingData* tilingData)
    {
        this->r_ = tilingData->r;
        this->a0_ = tilingData->a0;
        this->a1_ = tilingData->a1;

        this->blockNum_ = tilingData->blockNum;
        this->totalTiles_ = tilingData->totalTiles;
        this->tilesPerCore_ = tilingData->tilesPerCore;
        this->a0Outer_ = tilingData->a0Outer; 
        this->a0Inner_ = tilingData->a0Inner; 
        this->a0Tail_ = tilingData->a0Tail; 

        this->welfordTileLength_ = tilingData->welfordTileLength;
        this->welfordUpdateTimes_ = tilingData->welfordUpdateTimes;
        this->welfordUpdateTail_ = tilingData->welfordUpdateTail;   
        this->epsilon_ = tilingData->epsilon;
        this->welfordTempSize_ = tilingData->welfordTempSize;
        this->apiTempBufferSize_ = tilingData->apiTempBufferSize;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance)
    {
        // Init
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= blockNum_) {
            return;
        }
        processTileNum_ = tilesPerCore_;
        if (blockIdx_ == blockNum_ - 1) {
            processTileNum_ = totalTiles_ - (blockNum_ - 1) * tilesPerCore_;
        }

        aOffsetStart_ = blockIdx_ * tilesPerCore_;

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        gammaGm_.SetGlobalBuffer((__gm__ T_BETA*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ T_BETA*)beta);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        meanGm_.SetGlobalBuffer((__gm__ T_MEAN*)mean);
        varianceGm_.SetGlobalBuffer((__gm__ T_MEAN*)variance);

        // init local memory
        pipe_.InitBuffer(inQueueX_, DOUBLE_BUFFER, welfordTileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueY_, DOUBLE_BUFFER, welfordTileLength_ * sizeof(T));
        pipe_.InitBuffer(inQueueGamma_, DOUBLE_BUFFER, a0Inner_ * sizeof(T_BETA));
        pipe_.InitBuffer(inQueueBeta_, DOUBLE_BUFFER, a0Inner_ * sizeof(T_BETA));
        pipe_.InitBuffer(outQueueMean_, DOUBLE_BUFFER, a0Inner_ * sizeof(float));
        pipe_.InitBuffer(outQueueVariance_, DOUBLE_BUFFER, a0Inner_ * sizeof(float));

        pipe_.InitBuffer(welfordTempBuffer_, welfordTempSize_);
        pipe_.InitBuffer(apiTempBuffer_, apiTempBufferSize_);
        pipe_.InitBuffer(countTbuf, welfordTileLength_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        // Process
        if (blockIdx_ >= blockNum_) {
            return;
        }

        InitMeanVarianceLocal();
        mean_ = welfordTempBuffer_.Get<float>();
        variance_ = mean_[welfordTileLength_];
        rstdTensor = variance_[welfordTileLength_];
        shared_ = apiTempBuffer_.Get<uint8_t>();
        counts_ = countTbuf.Get<int32_t>();
        CaculateCountBuf(counts_);

        for (int64_t i = 0; i < processTileNum_; ++i) {
            uint64_t aOffsetNow = aOffsetStart_ + i; 
            uint64_t a1Offest = aOffsetNow % a1_;
            uint64_t a0Offest = (aOffsetNow / a1_) * a0Inner_;

            xyGmOffset = a1Offest * a0_ * r_ + a0Offest * r_;
            gammaGmOffset = a0Offest;
            meanGmOffset = a1Offest * a0_ + a0Offest;

            uint64_t a0ProcessNum = ((aOffsetNow / a1_) == a0Outer_ - 1) ? a0Tail_ : a0Inner_;

            if (i == 0 || a1Offest == 0) {
                if (i != 0) {
                    inQueueGamma_.FreeTensor(gammaTensor);
                    inQueueBeta_.FreeTensor(betaTensor);
                }
                CopyInGammaBetta(gammaGmOffset, a0ProcessNum);
            }

            for (int64_t j = 0; j < a0ProcessNum; j++) {
                // Welford Algorithm
                WelfordInitialize(mean_, variance_, welfordTileLength_);
                for (int64_t welfordUpdateCount = 0; welfordUpdateCount < welfordUpdateTimes_; welfordUpdateCount++) {
                    int64_t offset = xyGmOffset + welfordUpdateCount * welfordTileLength_;
                    ProcessWelfordUpdate(offset, welfordTileLength_);
                }
                if (welfordUpdateTail_ > 0) {
                    int64_t offset = xyGmOffset + welfordUpdateTimes_ * welfordTileLength_;
                    ProcessWelfordUpdate(offset, welfordUpdateTail_);
                }
                int64_t tailCountNum = welfordCount_ - (welfordUpdateTail_ > 0);
                int64_t tailCountLengthNum = welfordTileLength_ - welfordUpdateTail_;
                if (tailCountNum == 0) {
                    tailCountLengthNum = 0;
                }
                WelfordFinalizePara para;
                para.rnLength = welfordCount_;
                para.abLength = welfordTileLength_;
                para.headCount = welfordCount_;
                para.headCountLength = welfordUpdateTail_;
                para.tailCount = tailCountNum;
                para.tailCountLength = tailCountLengthNum;
                para.abRec = 1.0f / static_cast<float>(welfordTileLength_);
                para.rRec = 1.0f / static_cast<float>(r_);
                WelfordFinalize<true>(meanTensor[j], varianceTensor[j], mean_, variance_, counts_, shared_, para);

                CalculateBatchRstd(j);
                // Normalize
                for (int64_t welfordUpdateCount = 0; welfordUpdateCount < welfordUpdateTimes_; welfordUpdateCount++) {
                    int64_t fmOffset = xyGmOffset + welfordUpdateCount * welfordTileLength_;
                    ProcessNormalize(fmOffset, j, welfordTileLength_);
                }
                if (welfordUpdateTail_ > 0) {
                    int64_t fmOffset = xyGmOffset + welfordUpdateTimes_ * welfordTileLength_;
                    ProcessNormalize(fmOffset, j, welfordUpdateTail_);
                }

                xyGmOffset += r_;
            }
            CopyoutMeanVarience(a0ProcessNum);
            InitMeanVarianceLocal();
        }
        inQueueGamma_.FreeTensor(gammaTensor);
        inQueueBeta_.FreeTensor(betaTensor);
    }

private:
    __aicore__ inline void CaculateCountBuf(LocalTensor<int32_t>& tCountTensor)
    {
        __local_mem__ int32_t* tmpCountLocal = (__local_mem__ int32_t*)tCountTensor.GetPhyAddr();

        int64_t tailcoreProcessNum = welfordUpdateTimes_;
        int64_t formercoreProcessNum = welfordUpdateTimes_;
        uint32_t firstNum = welfordTileLength_;
        uint32_t secondNum = welfordTileLength_;
        if (welfordUpdateTail_ > 0) {
            tailcoreProcessNum = welfordUpdateTimes_;
            formercoreProcessNum = welfordUpdateTimes_ + 1;
            firstNum = welfordTileLength_;
            secondNum = welfordUpdateTail_;
        }

        int32_t tailCoreAddCount = static_cast<int32_t>(tailcoreProcessNum);
        int32_t formerCoreAddCount = static_cast<int32_t>(formercoreProcessNum);

        uint16_t fisrstLoopCount = CEIL_DIV(firstNum, VL_F32);
        uint16_t secondLoopCount = CEIL_DIV(secondNum, VL_F32);

        __VEC_SCOPE__
        {
            RegTensor<int32_t> tmpCount;
            MaskReg pregMain = CreateMask<int32_t, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg3 = firstNum;
            Duplicate(tmpCount, tailCoreAddCount, pregMain);
            for (uint16_t i = 0; i < fisrstLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<int32_t>(sreg3);
                DataCopy(((__local_mem__ int32_t*)tmpCountLocal + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg4 = secondNum;
            Duplicate(tmpCount, formerCoreAddCount, pregMain);
            for (uint16_t i = 0; i < secondLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<int32_t>(sreg4);
                DataCopy(((__local_mem__ int32_t*)tmpCountLocal + i * VL_F32), tmpCount, pregLoop);
            }

        }
    }

    __aicore__ inline void InitMeanVarianceLocal()
    {
        // ResetCache
        meanTensor = outQueueMean_.template AllocTensor<float>();
        varianceTensor = outQueueVariance_.template AllocTensor<float>();
    }

    __aicore__ inline void CastBatchMeanVariance(uint64_t currentANum)
    {
      __local_mem__ float* batchMeanInAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
      __local_mem__ float* batchVarianceInAddr = (__local_mem__ float*)varianceTensor.GetPhyAddr();
      __local_mem__ T_MEAN* batchMeanOutAddr = (__local_mem__ T_MEAN*)meanTensor.GetPhyAddr();
      __local_mem__ T_MEAN* batchVarianceOutAddr = (__local_mem__ T_MEAN*)varianceTensor.GetPhyAddr();

      uint32_t castCount = static_cast<uint32_t>(currentANum);
      uint16_t castLoops = static_cast<uint32_t>((castCount + VL_F32 - 1) / VL_F32);
      __VEC_SCOPE__
      {
        RegTensor<float> input_mean;
        RegTensor<float> input_variance;
        RegTensor<T_MEAN> output_mean;
        RegTensor<T_MEAN> output_variance;
        MicroAPI::MaskReg pregLoop;
        for (uint16_t i = 0; i < castLoops; i++) {
          pregLoop = MicroAPI::UpdateMask<float>(castCount);
          MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_mean, batchMeanInAddr + VL_F32 * i);
          MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_variance, batchVarianceInAddr + VL_F32 * i);
          Cast<T_MEAN, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
          Cast<T_MEAN, float, castTraitB322B16>(output_variance, input_variance, pregLoop);
          DataCopy<T_MEAN, StoreDist::DIST_PACK_B32>(((__local_mem__ T_MEAN *)batchMeanOutAddr + i * VL_MEAN), output_mean, pregLoop);
          DataCopy<T_MEAN, StoreDist::DIST_PACK_B32>(((__local_mem__ T_MEAN *)batchVarianceOutAddr + i * VL_MEAN), output_variance, pregLoop);
        }
      }
    }

    __aicore__ inline void CopyoutMeanVarience(const int64_t elemCnt) 
    {
      if constexpr(!IsSameType<T_MEAN, float>::value) {
        // float to bfloat16 or float16, input continue and output each repeat have only half value
        CastBatchMeanVariance(elemCnt);
        outQueueMean_.EnQue(meanTensor);
        outQueueVariance_.EnQue(varianceTensor);
        meanTensor = outQueueMean_.template DeQue<float>();
        varianceTensor = outQueueVariance_.template DeQue<float>();

        // VL_F32
        uint32_t castDmaCount = static_cast<uint32_t>(elemCnt);
        uint32_t castDmaLoops = static_cast<uint32_t>(castDmaCount / VL_F32);
        if (castDmaLoops > 0) {
          DataCopyExtParams copyInParams;
          copyInParams.blockCount = castDmaLoops;
          copyInParams.blockLen = VL_F32 * sizeof(T_MEAN);
          copyInParams.srcStride = (VECTOR_REG_WIDTH - VL_F32 * sizeof(T_MEAN)) / BLOCK_SIZE;
          copyInParams.dstStride = 0;
          DataCopyPad(meanGm_[meanGmOffset], meanTensor.ReinterpretCast<T_MEAN>(), copyInParams);
          DataCopyPad(varianceGm_[meanGmOffset], varianceTensor.ReinterpretCast<T_MEAN>(), copyInParams);
        }

        // tail
        uint32_t tailSize = static_cast<uint32_t>(castDmaCount % VL_F32);
        if (tailSize > 0) {
          DataCopyExtParams copyInParamsTail;
          copyInParamsTail.blockCount = 1;
          copyInParamsTail.blockLen = tailSize * sizeof(T_MEAN);
          copyInParamsTail.srcStride = 0;
          copyInParamsTail.dstStride = 0;
          DataCopyPad(meanGm_[meanGmOffset + castDmaLoops * VL_F32], meanTensor[castDmaLoops * VL_F32].ReinterpretCast<T_MEAN>(), copyInParamsTail);
          DataCopyPad(varianceGm_[meanGmOffset + castDmaLoops * VL_F32], varianceTensor[castDmaLoops * VL_F32].ReinterpretCast<T_MEAN>(), copyInParamsTail);
        }
        outQueueMean_.FreeTensor(meanTensor);
        outQueueVariance_.FreeTensor(varianceTensor);
      } else {
        // Refresh Cache
        outQueueMean_.EnQue(meanTensor);
        meanTensor = outQueueMean_.template DeQue<float>();
        CopyOut(meanGm_[meanGmOffset], meanTensor, elemCnt);
        outQueueMean_.FreeTensor(meanTensor);

        outQueueVariance_.EnQue(varianceTensor);
        varianceTensor = outQueueVariance_.template DeQue<float>();
        CopyOut(varianceGm_[meanGmOffset], varianceTensor, elemCnt);
        outQueueVariance_.FreeTensor(varianceTensor);
      }
    }

    __aicore__ inline void WelfordInitialize(const LocalTensor<float>& mean, const LocalTensor<float>& variance,
                                             const int64_t elemCnt)
    {
        // WelfordInitialize
        welfordCount_ = 0;
        constexpr static uint32_t VL_B32 = platform::GetVRegSize() / sizeof(float);
        uint16_t loopTimes = (elemCnt + VL_B32 - 1) / VL_B32;
        __VEC_SCOPE__
        {
            __local_mem__ float* meamPtr = (__local_mem__ float*)mean.GetPhyAddr();
            __local_mem__ float* variancePtr = (__local_mem__ float*)variance.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(elemCnt);
            AscendC::MicroAPI::RegTensor<float> xReg;
            AscendC::MicroAPI::MaskReg pMask;
            Duplicate(xReg, 0.0f);
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                DataCopy((__local_mem__ float*)meamPtr + i * VL_B32, xReg, pMask);
                DataCopy((__local_mem__ float*)variancePtr + i * VL_B32, xReg, pMask);
            }
        }
    }

    __aicore__ inline void ProcessWelfordUpdate(const int64_t offset, const int64_t elemCnt)
    {
        // ProcessWelfordUpdate
        welfordCount_++;
        xTensor = inQueueX_.template AllocTensor<T>();
        CopyIn(xTensor, xGm_[offset], elemCnt);
        inQueueX_.EnQue(xTensor);
        xTensor = inQueueX_.template DeQue<T>();

        WelfordUpdateParam para;
        para.rnLength = 1;
        para.abLength = welfordTileLength_;
        para.abComputeLength = elemCnt;
        para.nRec = 1.0f / static_cast<float>(welfordCount_);
        WelfordUpdate<T, float, false>(mean_, variance_, mean_, variance_, xTensor, shared_, para);

        inQueueX_.FreeTensor(xTensor);
    }

    __aicore__ inline void CalculateBatchRstd(const int64_t offset)
    {
        __local_mem__ float* batchRstdTensorTensorAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ float* batchVarTensorTensorAddr = (__local_mem__ float*)varianceTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> one;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> e;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> rstd;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(one, 1.0, pregOne);

            DataCopy<float, LoadDist::DIST_BRC_B32>(var, ((__local_mem__ float*)batchVarTensorTensorAddr + offset));
            Duplicate(scalar1, float(0.5), pregOne);
            Duplicate(scalarInf, POS_INF, pregOne);
            Duplicate(scalarZero, float(0.0), pregOne);
            Duplicate(t1, float(1.5), pregOne);
            Duplicate(s, float(1.0), pregOne);
            Adds(var, var, epsilon_, pregOne);
            Div(r, one, var, pregOne);
            Sqrt(y, r, pregOne);
            Muls(t, var, float(-0.5), pregOne);
            Mul(t, t, y, pregOne);                // -0.5 * x * y
            Mula(t1, t, y, pregOne);              // 1.5 + (-0.5 * x * y) * y
            Mul(rstd, y, t1, pregOne);            // y = y * (1.5 - 0.5 * x * y)
            Muls(t2, var, float(-1.0), pregOne);  // -1 * x
            Mula(s, t2, r, pregOne);              // 1 + (-1) * x * r
            Muls(t3, rstd, float(-1.0), pregOne); // (-1) * y
            Mula(r, t3, rstd, pregOne);           // r + (-1) * y * y
            Mula(s, var, r, pregOne);             // s + x * t
            Mul(s, s, rstd, pregOne);             // e * y
            Mula(rstd, s, scalar1, pregOne);      // y + y * e * 0.5
            CompareScalar(cmpRegZero, var, POS_INF, pregOne);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            CompareScalar(cmpRegInf, var, float(0.0), pregOne);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)batchRstdTensorTensorAddr + offset), rstd, pregOne);
        }
    }

    __aicore__ inline void CopyInGammaBetta(const int64_t paramOffset, const int64_t elemCnt) 
    {
        gammaTensor = inQueueGamma_.template AllocTensor<T_BETA>();
        CopyIn<T_BETA>(gammaTensor, gammaGm_[paramOffset], elemCnt);
        inQueueGamma_.EnQue(gammaTensor);
        gammaTensor = inQueueGamma_.template DeQue<T_BETA>();

        betaTensor = inQueueBeta_.template AllocTensor<T_BETA>();
        CopyIn<T_BETA>(betaTensor, betaGm_[paramOffset], elemCnt);
        inQueueBeta_.EnQue(betaTensor);
        betaTensor = inQueueBeta_.template DeQue<T_BETA>();

    }

    template <typename T_SRC>
    __aicore__ inline void LoadTensorForDtypeT(
        RegTensor<float>& dst, __local_mem__ T_SRC* input, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
        }
    }

    template <typename T_SRC_GAMMA>
    __aicore__ inline void LoadOneNumberTensorForDtypeT(
        RegTensor<float>& dst, __local_mem__ T_SRC_GAMMA* input, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC_GAMMA, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<T_SRC_GAMMA, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst, ((__local_mem__ float*)(input) + (offset)));
        }
    }

    __aicore__ inline void CalY(const int64_t elemCnt, const int64_t gammaUbOffset)
    {
        __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ T* xTensorAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ T* yTensorAddr = (__local_mem__ T*)yTensor.GetPhyAddr();
        __local_mem__ T_BETA* gammaTensorAddr = (__local_mem__ T_BETA*)gammaTensor.GetPhyAddr();
        __local_mem__ T_BETA* betaTensorAddr = (__local_mem__ T_BETA*)betaTensor.GetPhyAddr();

        uint16_t numLoop = CEIL_DIV(elemCnt, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> y;
            DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)batchMeanTensorAddr + gammaUbOffset));
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, ((__local_mem__ float*)batchRstdTensorAddr + gammaUbOffset));

            MaskReg mask0;
            uint32_t sreg0 = elemCnt;
            for (uint16_t i = 0; i < numLoop; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadOneNumberTensorForDtypeT<T_BETA>(gamma, gammaTensorAddr, mask0, gammaUbOffset);
                LoadOneNumberTensorForDtypeT<T_BETA>(beta, betaTensorAddr, mask0, gammaUbOffset);

                LoadTensorForDtypeT(x1, xTensorAddr, mask0, i * VL_F32);
                Sub(x1, x1, mean, mask0);
                Mul(x1, x1, rstd, mask0);
                Mul(x1, x1, gamma, mask0);
                Add(y, x1, beta, mask0);
                if constexpr (IsSameType<T, half>::value) {
                    RegTensor<half> yFp16;
                    Cast<half, float, castTraitB322B16>(yFp16, y, mask0);
                    DataCopy<half, StoreDist::DIST_PACK_B32>(
                        yTensorAddr + i * VL_F32, yFp16, mask0);
                } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                    RegTensor<bfloat16_t> xBf16;
                    Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y, mask0);
                    DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                        yTensorAddr + i * VL_F32, xBf16, mask0);
                } else {
                    DataCopy(yTensorAddr + i * VL_F32, y, mask0);
                }

            }
        
        }
    }

    __aicore__ inline void ProcessNormalize(const int64_t fmOffset, const int64_t gammaUbOffset, const int64_t elemCnt)
    {
        // ProcessNormalize
        xTensor = inQueueX_.template AllocTensor<T>();
        CopyIn(xTensor, xGm_[fmOffset], elemCnt);
        inQueueX_.EnQue(xTensor);
        xTensor = inQueueX_.template DeQue<T>();

        yTensor = outQueueY_.template AllocTensor<T>();

        CalY(elemCnt, gammaUbOffset);
    
        inQueueX_.FreeTensor(xTensor);

        outQueueY_.EnQue(yTensor);
        yTensor = outQueueY_.template DeQue<T>();
        CopyOut(yGm_[fmOffset], yTensor, elemCnt);
        outQueueY_.FreeTensor(yTensor);
    }

private:
    // Constants
    constexpr static int64_t DOUBLE_BUFFER = 2;
    // constexpr static int64_t AGGREGATION_COUNT = 256;
    constexpr static uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    constexpr static uint32_t VL_MEAN = VECTOR_REG_WIDTH / sizeof(T_MEAN);
    constexpr static int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static float POS_INF = 3.40282366920938E+38;

    TPipe pipe_;

    // GM Buffer
    // global memory address 
    GlobalTensor<T> xGm_;
    GlobalTensor<T_BETA> gammaGm_;
    GlobalTensor<T_BETA> betaGm_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T_MEAN> meanGm_;
    GlobalTensor<T_MEAN> varianceGm_;

    // Local Tensor
    LocalTensor<T> xTensor;
    LocalTensor<T> yTensor;
    LocalTensor<T_BETA> gammaTensor;
    LocalTensor<T_BETA> betaTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> varianceTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<float> mean_;
    LocalTensor<float> variance_;
    LocalTensor<uint8_t> shared_;
    LocalTensor<int32_t> counts_;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueBeta_;

    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TQue<QuePosition::VECOUT, 1> outQueueMean_;
    TQue<QuePosition::VECOUT, 1> outQueueVariance_;

    TBuf<TPosition::VECCALC> welfordTempBuffer_;
    TBuf<TPosition::VECCALC> apiTempBuffer_;
    TBuf<TPosition::VECCALC> countTbuf;

    // Tiling data
    uint64_t r_{0};
    uint64_t a1_{0};
    uint64_t a0_{0};

    uint64_t blockNum_{0};
    uint64_t totalTiles_{0};
    uint64_t tilesPerCore_{0};
    uint64_t a0Outer_{0};
    uint64_t a0Inner_{0};
    uint64_t a0Tail_{0};

    uint64_t welfordTileLength_{0};
    uint64_t welfordUpdateTimes_{0};
    uint64_t welfordUpdateTail_{0};

    uint64_t welfordTempSize_{0};
    uint64_t apiTempBufferSize_{0};
    float epsilon_{0};

    uint32_t blockIdx_ = 0;
    uint64_t aOffsetStart_ = 0;
    uint64_t processTileNum_ = 0;
    uint64_t xyGmOffset = 0;
    uint64_t meanGmOffset = 0;
    uint64_t gammaGmOffset = 0;

    // params
    int64_t welfordCount_{0};
};
} // namespace InstanceNormOps
#endif // INSTANCE_NORM_AR_WELFORD_H_
