/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_quant_v2_regbase_recompute.h
 * \brief
 */
#ifndef RMS_NORM_QUANT_V2_REBASE_REDUCE_H_
#define RMS_NORM_QUANT_V2_REBASE_REDUCE_H_

#include "rms_norm_quant_v2_regbase_common.h"

namespace RmsNormQuantV2 {
// T_X:          x, gamma, beta
// yCopyDtype:   y1, y2
// T_SCALES:     scales1, scales2
// T_ZEROPOINTS: zero_points1, zero_points2
template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS>
class RmsNormQuantV2RegbaseRecompute {
    using yCopyDtype = std::conditional_t<IsSameType<T_Y, int4b_t>::value, uint8_t, T_Y>;

public:
    __aicore__ inline RmsNormQuantV2RegbaseRecompute(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1, GM_ADDR zeroPoints2,
        GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, const RmsNormQuantV2RegbaseRecomputeTilingData* tilingData)
    {
        numM_ = tilingData->numM;
        numN_ = tilingData->numN;
        baseM_ = tilingData->baseM;
        baseN_ = tilingData->baseN;

        mPerCore_ = tilingData->mPerCore;
        mLastCore_ = tilingData->mLastCore;
        nUbLoops_ = tilingData->nUbLoops;
        binAddQuotient_ = tilingData->binAddQuotient;
        powerSplit_ = tilingData->powerSplit;
        resultCacheID_ = GetCacheId(powerSplit_ - NUM_ONE); // ub间二分累加结果存储位置

        mainFoldCount_ = tilingData->mainFoldCount;
        foldTail_ = tilingData->foldTail;

        option_mask_ = tilingData->optionMask;
        div_mode_ = tilingData->divMode != 0;
        dst_type_ = tilingData->dstDtype;

        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;

        mCurCore_ = GetBlockIdx() == (GetBlockNum() - 1) ? mLastCore_ : mPerCore_;

        isHasScale2_ = option_mask_ & (1 << NUM_ZERO);
        isHasZeroPoint1_ = option_mask_ & (1 << NUM_ONE);
        isHasZeroPoint2_ = option_mask_ & (1 << NUM_TWO);
        isHasBeta_ = option_mask_ & (1 << NUM_THREE);
        isNeedBrc_ = option_mask_ & (1 << NUM_FOUR);
        option_mask_ = option_mask_ & 0xF;

        /// GlobalTensor Init
        int64_t gmOffset = GetBlockIdx() * mPerCore_ * numN_;
        int64_t ygmOffset = gmOffset;
        int64_t yLen = mCurCore_ * numN_;
        int64_t yQueueSize = baseN_ * sizeof(yCopyDtype);
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            ygmOffset /= NUM_TWO;
            yLen /= NUM_TWO;
            yQueueSize /= NUM_TWO;
        }
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, mCurCore_ * numN_);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        scales1Gm_.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ yCopyDtype*)y1 + ygmOffset, yLen);
        y2Gm_.SetGlobalBuffer((__gm__ yCopyDtype*)y2 + ygmOffset, yLen);
        /// Ub buffer init
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueXFold_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueScales1_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_SCALES));
        pipe_->InitBuffer(outQueueY1_, DOUBLE_BUFFER_NUM, yQueueSize);

        // optional
        isHasY2_ = isHasScale2_;
        if (isHasScale2_) {
            scales2Gm_.SetGlobalBuffer((__gm__ T_SCALES*)scales2, numN_);
            pipe_->InitBuffer(inQueueScales2_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_SCALES));
        }
        if (isHasZeroPoint1_) {
            zeroPoints1Gm_.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints1, numN_);
            pipe_->InitBuffer(inQueueZeroPoints1_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_ZEROPOINTS));
        }
        if (isHasZeroPoint2_) {
            zeroPoints2Gm_.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints2, numN_);
            pipe_->InitBuffer(inQueueZeroPoints2_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_ZEROPOINTS));
        }
        if (isHasBeta_) {
            betaGm_.SetGlobalBuffer((__gm__ T_X*)beta, numN_);
            pipe_->InitBuffer(inQueueBeta_, DOUBLE_BUFFER_NUM, baseN_ * sizeof(T_X));
        }
        if (isHasY2_) {
            pipe_->InitBuffer(outQueueY2_, DOUBLE_BUFFER_NUM, yQueueSize);
        }

        // tmp buffer allocate
        pipe_->InitBuffer(rstdBuf_, Aligned(static_cast<int64_t>(baseM_ * sizeof(float)), BLOCK_SIZE));
        pipe_->InitBuffer(
            cacheBuf_,
            Aligned(static_cast<int64_t>((resultCacheID_ + NUM_ONE) * sizeof(float)) * AR_RECOMPUTE_SUM_LEN, BLOCK_SIZE));
        pipe_->InitBuffer(binaryAddBuf_, Aligned(static_cast<int64_t>(VL_FP32 * NUM_TWO * sizeof(float)), BLOCK_SIZE));
        pipe_->InitBuffer(xFp32TmpBuf_, baseN_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t mCnt = CeilDiv(mCurCore_, baseM_);
        for (int64_t i = 0; i < mCnt; ++i) {
            uint32_t curM = (i == mCnt - 1) ? (mCurCore_ - (mCnt - 1) * baseM_) : baseM_;
            LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
            for (uint32_t j = 0; j < curM; ++j) {
                // 逐行 ub间二分累加
                int64_t xGmOffset = (i * baseM_ + j) * numN_;
                ComputeOneLineXSquareSum(rstdLocal, xGmOffset, j);
            }
            // 计算 rstd
            CalculateRstd(rstdLocal, rstdLocal, curM, avgFactor_, epsilon_);
            // 计算 Y
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyParams xDataCopyParams;
            xDataCopyParams.blockCount = 1;
            xDataCopyParams.srcStride = 0;
            xDataCopyParams.dstStride = 0;

            LocalTensor<T_SCALES> scales1Local;
            // optional input
            LocalTensor<T_SCALES> scales2Local;
            LocalTensor<T_ZEROPOINTS> zeroPoints1Local, zeroPoints2Local;
            if (isNeedBrc_) {
                CopyInQuant(0, 1);
                scales1Local = inQueueScales1_.DeQue<T_SCALES>();
                if (isHasScale2_) {
                    scales2Local = inQueueScales2_.DeQue<T_SCALES>();
                }
                if (isHasZeroPoint1_) {
                    zeroPoints1Local = inQueueZeroPoints1_.DeQue<T_ZEROPOINTS>();
                }
                if (isHasZeroPoint2_) {
                    zeroPoints2Local = inQueueZeroPoints2_.DeQue<T_ZEROPOINTS>();
                }
            }

            for (uint32_t j = 0; j < nUbLoops_; ++j) { // 先循环 r 轴
                uint32_t curN = (j == nUbLoops_ - 1) ? (numN_ - (nUbLoops_ - 1) * baseN_) : baseN_;
                CopyInRPattern<T_X>(inQueueGamma_, gammaGm_, j * baseN_, curN);
                if (!isNeedBrc_) {
                    CopyInQuant(j * baseN_, curN);
                }
                if (isHasBeta_) {
                    CopyInRPattern<T_X>(inQueueBeta_, betaGm_, j * baseN_, curN);
                }
                if (!isNeedBrc_) {
                    scales1Local = inQueueScales1_.DeQue<T_SCALES>();
                    if (isHasScale2_) {
                        scales2Local = inQueueScales2_.DeQue<T_SCALES>();
                    }
                    if (isHasZeroPoint1_) {
                        zeroPoints1Local = inQueueZeroPoints1_.DeQue<T_ZEROPOINTS>();
                    }
                    if (isHasZeroPoint2_) {
                        zeroPoints2Local = inQueueZeroPoints2_.DeQue<T_ZEROPOINTS>();
                    }
                }
                LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
                LocalTensor<T_X> betaLocal;

                if (isHasBeta_) {
                    betaLocal = inQueueBeta_.DeQue<T_X>();
                }
                for (int64_t k = 0; k < curM; ++k) {
                    int64_t xGmOffset = (i * baseM_ + k) * numN_ + j * baseN_;
                    xDataCopyParams.blockLen = curN * sizeof(T_X);
                    LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
                    DataCopyPad(xLocal, xGm_[xGmOffset], xDataCopyParams, padParams);
                    inQueueX_.EnQue<T_X>(xLocal);
                    xLocal = inQueueX_.DeQue<T_X>();

                    if (option_mask_ == 0b0000) {
                        ComputeY<false, false, false, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1000) {
                        ComputeY<false, false, false, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0100) {
                        ComputeY<false, false, true, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1100) {
                        ComputeY<false, false, true, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0010) {
                        ComputeY<false, true, false, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1010) {
                        ComputeY<false, true, false, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0110) {
                        ComputeY<false, true, true, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1110) {
                        ComputeY<false, true, true, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0001) {
                        ComputeY<true, false, false, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1001) {
                        ComputeY<true, false, false, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0101) {
                        ComputeY<true, false, true, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1101) {
                        ComputeY<true, false, true, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0011) {
                        ComputeY<true, true, false, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1011) {
                        ComputeY<true, true, false, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b0111) {
                        ComputeY<true, true, true, false>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    } else if (option_mask_ == 0b1111) {
                        ComputeY<true, true, true, true>(
                            xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                            zeroPoints2Local, betaLocal, xGmOffset, curN, k);
                    }
                    inQueueX_.FreeTensor(xLocal);
                    // 输出
                    CopyOutY(outQueueY1_, y1Gm_, xGmOffset, curN);
                    if (isHasY2_) {
                        CopyOutY(outQueueY2_, y2Gm_, xGmOffset, curN);
                    }
                }
                inQueueGamma_.FreeTensor(gammaLocal);
                if (isHasBeta_) {
                    inQueueBeta_.FreeTensor(betaLocal);
                }
                if (!isNeedBrc_) {
                    inQueueScales1_.FreeTensor(scales1Local);
                    if (isHasScale2_) {
                        inQueueScales2_.FreeTensor(scales2Local);
                    }
                    if (isHasZeroPoint1_) {
                        inQueueZeroPoints1_.FreeTensor(zeroPoints1Local);
                    }
                    if (isHasZeroPoint2_) {
                        inQueueZeroPoints2_.FreeTensor(zeroPoints2Local);
                    }
                }
            }
            if (isNeedBrc_) {
                inQueueScales1_.FreeTensor(scales1Local);
                if (isHasScale2_) {
                    inQueueScales2_.FreeTensor(scales2Local);
                }
                if (isHasZeroPoint1_) {
                    inQueueZeroPoints1_.FreeTensor(zeroPoints1Local);
                }
                if (isHasZeroPoint2_) {
                    inQueueZeroPoints2_.FreeTensor(zeroPoints2Local);
                }
            }
        }
    }

    __aicore__ inline void ComputeOneLineXSquareSum(LocalTensor<float>& rstdLocal, int64_t offset, uint32_t rowIndex)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams xDataCopyParams;
        xDataCopyParams.blockCount = 1;
        xDataCopyParams.srcStride = 0;
        xDataCopyParams.dstStride = 0;
        DataCopyParams xFoldDataCopyParams;
        xFoldDataCopyParams.blockCount = 1;
        xFoldDataCopyParams.srcStride = 0;
        xFoldDataCopyParams.dstStride = 0;
        LocalTensor<float> cacheLocal = cacheBuf_.Get<float>(); // ub间二分累加缓存结果
        LocalTensor<float> xFp32Tmp = xFp32TmpBuf_.Get<float>();

        // ub间二分累加
        for (int64_t r = 0; r < powerSplit_; ++r) {
            int64_t xGmOffset1 = offset + baseN_ * r;
            int64_t xGmOffset2 = offset + baseN_ * (r + powerSplit_);

            xDataCopyParams.blockLen = baseN_ * sizeof(T_X);
            LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
            DataCopyPad(xLocal, xGm_[xGmOffset1], xDataCopyParams, padParams);
            inQueueX_.EnQue<T_X>(xLocal);

            xLocal = inQueueX_.DeQue<T_X>();
            if (r < mainFoldCount_) {
                xFoldDataCopyParams.blockLen = baseN_ * sizeof(T_X);
                LocalTensor<T_X> xFoldLocal = inQueueXFold_.AllocTensor<T_X>();
                DataCopyPad(xFoldLocal, xGm_[xGmOffset2], xFoldDataCopyParams, padParams);
                inQueueXFold_.EnQue<T_X>(xFoldLocal);
                xFoldLocal = inQueueXFold_.DeQue<T_X>();
                FoldBlockVF(xLocal, xFoldLocal, xFp32Tmp, baseN_, baseN_);
                inQueueXFold_.FreeTensor(xFoldLocal);
            } else if (r == mainFoldCount_ && foldTail_ > 0) {
                xFoldDataCopyParams.blockLen = foldTail_ * sizeof(T_X);
                LocalTensor<T_X> xFoldLocal = inQueueXFold_.AllocTensor<T_X>();
                DataCopyPad(xFoldLocal, xGm_[xGmOffset2], xFoldDataCopyParams, padParams);
                inQueueXFold_.EnQue<T_X>(xFoldLocal);
                xFoldLocal = inQueueXFold_.DeQue<T_X>();
                FoldBlockVF(xLocal, xFoldLocal, xFp32Tmp, foldTail_, baseN_);
                inQueueXFold_.FreeTensor(xFoldLocal);
            } else { // cast 输入到 fp32
                if constexpr (IsSameType<T_X, float>::value) {
                    Mul<float>(xFp32Tmp, xLocal, xLocal, baseN_);
                } else {
                    Cast<float, T_X>(xFp32Tmp, xLocal, AscendC::RoundMode::CAST_NONE, baseN_);
                    Mul<float>(xFp32Tmp, xFp32Tmp, xFp32Tmp, baseN_);
                }
            }
            CalculateSquareReduceSum(xFp32Tmp, xFp32Tmp, baseN_); // 整块 ub 二分累加
            int64_t cacheId = GetCacheId(r);
            UpdateCache(cacheLocal, xFp32Tmp, cacheId, AR_RECOMPUTE_SUM_LEN);
            inQueueX_.FreeTensor(xLocal);
        }

        // 输出转连续  ub 到 ub 搬运
        __local_mem__ float* dstPtr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* cachePtr =
            (__local_mem__ float*)cacheLocal.GetPhyAddr() + resultCacheID_ * AR_RECOMPUTE_SUM_LEN;
        __VEC_SCOPE__
        {
            RegTensor<float> a;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_NORM>(a, cachePtr);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + rowIndex, a, pregOne);
        }
    }

    __aicore__ inline void UpdateCache(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheId,
        const int64_t stride)
    {
        uint16_t innerLoopTimes = cacheId;
        uint32_t innerLoopStride = stride;
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> aReg, bReg;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            DataCopy(aReg, (__local_mem__ float*)src);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, dst + j * innerLoopStride);
                Add(aReg, aReg, bReg, pregOne);
            }
            DataCopy((__local_mem__ float*)cache, aReg, pregOne);
        }
    }

    template <bool HAS_SCALES2, bool HAS_ZEROPOINTS1, bool HAS_ZEROPOINTS2, bool HAS_BETA>
    __aicore__ inline void ComputeY(
        LocalTensor<T_X>& xLocal, LocalTensor<float>& rstdLocal, LocalTensor<T_X>& gammaLocal,
        LocalTensor<T_SCALES>& scales1Local, LocalTensor<T_SCALES>& scales2Local,
        LocalTensor<T_ZEROPOINTS>& zeroPoints1Local, LocalTensor<T_ZEROPOINTS>& zeroPoints2Local,
        LocalTensor<T_X>& betaLocal, int64_t gmOffset, uint32_t curN, uint32_t mIdx)
    {
        LocalTensor<yCopyDtype> y1Local = outQueueY1_.AllocTensor<yCopyDtype>();
        LocalTensor<yCopyDtype> y2Local;
        if constexpr (HAS_SCALES2) {
            y2Local = outQueueY2_.AllocTensor<yCopyDtype>();
        }

        if (div_mode_) {
            if (isNeedBrc_) {
                ComputeY_VF<HAS_SCALES2, HAS_ZEROPOINTS1, HAS_ZEROPOINTS2, HAS_BETA, true, true>(
                    y1Local, y2Local, xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                    zeroPoints2Local, betaLocal, mIdx, curN);
            } else {
                ComputeY_VF<HAS_SCALES2, HAS_ZEROPOINTS1, HAS_ZEROPOINTS2, HAS_BETA, true, false>(
                    y1Local, y2Local, xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                    zeroPoints2Local, betaLocal, mIdx, curN);
            }
        } else {
            if (isNeedBrc_) {
                ComputeY_VF<HAS_SCALES2, HAS_ZEROPOINTS1, HAS_ZEROPOINTS2, HAS_BETA, false, true>(
                    y1Local, y2Local, xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                    zeroPoints2Local, betaLocal, mIdx, curN);
            } else {
                ComputeY_VF<HAS_SCALES2, HAS_ZEROPOINTS1, HAS_ZEROPOINTS2, HAS_BETA, false, false>(
                    y1Local, y2Local, xLocal, rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local,
                    zeroPoints2Local, betaLocal, mIdx, curN);
            }
        }
        outQueueY1_.EnQue<yCopyDtype>(y1Local);
        if constexpr (HAS_SCALES2) {
            outQueueY2_.EnQue<yCopyDtype>(y2Local);
        }
    }

    template <bool HAS_SCALES2, bool HAS_ZEROPOINTS1, bool HAS_ZEROPOINTS2, bool HAS_BETA, bool DIV_MODE, bool NEED_BRC>
    __aicore__ inline void ComputeY_VF(
        LocalTensor<yCopyDtype>& y1Local, LocalTensor<yCopyDtype>& y2Local, LocalTensor<T_X>& xLocal,
        LocalTensor<float>& rstdLocal, LocalTensor<T_X>& gammaLocal, LocalTensor<T_SCALES>& scales1Local,
        LocalTensor<T_SCALES>& scales2Local, LocalTensor<T_ZEROPOINTS>& zeroPoints1Local,
        LocalTensor<T_ZEROPOINTS>& zeroPoints2Local, LocalTensor<T_X>& betaLocal, uint32_t rstdOffset, uint32_t count)
    {
        uint32_t sreg = (uint32_t)count;
        uint16_t repeatTimes = CeilDivision(count, VL_FP32);
        __local_mem__ T_X* xAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_X* gammaAddr = (__local_mem__ T_X*)gammaLocal.GetPhyAddr();

        __local_mem__ T_SCALES* scales1Addr = (__local_mem__ T_SCALES*)scales1Local.GetPhyAddr();
        __local_mem__ T_SCALES* scales2Addr;
        if constexpr (HAS_SCALES2) {
            scales2Addr = (__local_mem__ T_SCALES*)scales2Local.GetPhyAddr();
        }
        __local_mem__ T_ZEROPOINTS* zeroPoints1Addr;
        __local_mem__ T_ZEROPOINTS* zeroPoints2Addr;
        if constexpr (HAS_ZEROPOINTS1) {
            zeroPoints1Addr = (__local_mem__ T_ZEROPOINTS*)zeroPoints1Local.GetPhyAddr();
        }
        if constexpr (HAS_ZEROPOINTS2) {
            zeroPoints2Addr = (__local_mem__ T_ZEROPOINTS*)zeroPoints2Local.GetPhyAddr();
        }
        __local_mem__ T_X* betaAddr;
        if constexpr (HAS_BETA) {
            betaAddr = (__local_mem__ T_X*)betaLocal.GetPhyAddr();
        }
        __local_mem__ yCopyDtype* y1Addr = (__local_mem__ yCopyDtype*)y1Local.GetPhyAddr();
        __local_mem__ yCopyDtype* y2Addr;
        if constexpr (HAS_SCALES2) {
            y2Addr = (__local_mem__ yCopyDtype*)y2Local.GetPhyAddr();
        }

        if constexpr (NEED_BRC) {
            __VEC_SCOPE__
            {
                RegTensor<float> xRegFp32, gammaRegFp32, rstdReg, betaRegFp32;
                RegTensor<float> scales1RegFp32, zeroPoints1RegFp32;
                RegTensor<float> scales2RegFp32, zeroPoints2RegFp32;
                RegTensor<float> y1Reg, y2Reg;
                MaskReg maskReg;
                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();
                LoadScalarForDtypeTIn<T_SCALES>(scales1Addr, scales1RegFp32, pregFull, 0);
                if constexpr (HAS_SCALES2) {
                    LoadScalarForDtypeTIn<T_SCALES>(scales2Addr, scales2RegFp32, pregFull, 0);
                }
                if constexpr (HAS_ZEROPOINTS1) {
                    LoadScalarForDtypeTIn<T_ZEROPOINTS>(zeroPoints1Addr, zeroPoints1RegFp32, pregFull, 0);
                }
                if constexpr (HAS_ZEROPOINTS2) {
                    LoadScalarForDtypeTIn<T_ZEROPOINTS>(zeroPoints2Addr, zeroPoints2RegFp32, pregFull, 0);
                }
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadTensorForDtypeTIn<T_X>(xAddr, xRegFp32, maskReg, i * VL_FP32);
                    LoadTensorForDtypeTIn<T_X>(gammaAddr, gammaRegFp32, maskReg, i * VL_FP32);
                    if constexpr (HAS_BETA) {
                        LoadTensorForDtypeTIn<T_X>(betaAddr, betaRegFp32, maskReg, i * VL_FP32);
                    }
                    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                    if constexpr (HAS_BETA) {
                        Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                    }
                    if constexpr (DIV_MODE) {
                        Div(y1Reg, xRegFp32, scales1RegFp32, maskReg);
                    } else {
                        Mul(y1Reg, xRegFp32, scales1RegFp32, maskReg);
                    }
                    if constexpr (HAS_SCALES2) {
                        if constexpr (DIV_MODE) {
                            Div(y2Reg, xRegFp32, scales2RegFp32, maskReg);
                        } else {
                            Mul(y2Reg, xRegFp32, scales2RegFp32, maskReg);
                        }
                    }
                    if constexpr (HAS_ZEROPOINTS1) {
                        Add(y1Reg, y1Reg, zeroPoints1RegFp32, maskReg);
                    }
                    if constexpr (HAS_ZEROPOINTS2) {
                        Add(y2Reg, y2Reg, zeroPoints2RegFp32, maskReg);
                    }
                    StoreTensorForDtypeTOut<yCopyDtype>(y1Addr, y1Reg, maskReg, mask4Int4, i * VL_FP32);
                    if constexpr (HAS_SCALES2) {
                        StoreTensorForDtypeTOut<yCopyDtype>(y2Addr, y2Reg, maskReg, mask4Int4, i * VL_FP32);
                    }
                }
            }
        } else {
            __VEC_SCOPE__
            {
                RegTensor<float> xRegFp32, gammaRegFp32, rstdReg, betaRegFp32;
                RegTensor<float> scales1RegFp32, zeroPoints1RegFp32;
                RegTensor<float> scales2RegFp32, zeroPoints2RegFp32;
                RegTensor<float> y1Reg, y2Reg;
                MaskReg maskReg;
                MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadTensorForDtypeTIn<T_X>(xAddr, xRegFp32, maskReg, i * VL_FP32);
                    LoadTensorForDtypeTIn<T_X>(gammaAddr, gammaRegFp32, maskReg, i * VL_FP32);
                    LoadTensorForDtypeTIn<T_SCALES>(scales1Addr, scales1RegFp32, maskReg, i * VL_FP32);
                    if constexpr (HAS_SCALES2) {
                        LoadTensorForDtypeTIn<T_SCALES>(scales2Addr, scales2RegFp32, maskReg, i * VL_FP32);
                    }
                    if constexpr (HAS_ZEROPOINTS1) {
                        LoadTensorForDtypeTIn<T_ZEROPOINTS>(zeroPoints1Addr, zeroPoints1RegFp32, maskReg, i * VL_FP32);
                    }
                    if constexpr (HAS_ZEROPOINTS2) {
                        LoadTensorForDtypeTIn<T_ZEROPOINTS>(zeroPoints2Addr, zeroPoints2RegFp32, maskReg, i * VL_FP32);
                    }
                    if constexpr (HAS_BETA) {
                        LoadTensorForDtypeTIn<T_X>(betaAddr, betaRegFp32, maskReg, i * VL_FP32);
                    }
                    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                    if constexpr (HAS_BETA) {
                        Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                    }
                    if constexpr (DIV_MODE) {
                        Div(y1Reg, xRegFp32, scales1RegFp32, maskReg);
                    } else {
                        Mul(y1Reg, xRegFp32, scales1RegFp32, maskReg);
                    }
                    if constexpr (HAS_SCALES2) {
                        if constexpr (DIV_MODE) {
                            Div(y2Reg, xRegFp32, scales2RegFp32, maskReg);
                        } else {
                            Mul(y2Reg, xRegFp32, scales2RegFp32, maskReg);
                        }
                    }
                    if constexpr (HAS_ZEROPOINTS1) {
                        Add(y1Reg, y1Reg, zeroPoints1RegFp32, maskReg);
                    }
                    if constexpr (HAS_ZEROPOINTS2) {
                        Add(y2Reg, y2Reg, zeroPoints2RegFp32, maskReg);
                    }
                    StoreTensorForDtypeTOut<yCopyDtype>(y1Addr, y1Reg, maskReg, mask4Int4, i * VL_FP32);
                    if constexpr (HAS_SCALES2) {
                        StoreTensorForDtypeTOut<yCopyDtype>(y2Addr, y2Reg, maskReg, mask4Int4, i * VL_FP32);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void CopyOutY(
        TQue<QuePosition::VECOUT, 1>& outQueue, GlobalTensor<yCopyDtype>& tGm, int64_t offset, uint32_t len)
    {
        uint32_t copySize = len * sizeof(yCopyDtype);
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            copySize /= 2;
            offset /= 2;
        }
        LocalTensor<yCopyDtype> yLocal = outQueue.DeQue<yCopyDtype>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),        // blockCount
            static_cast<uint32_t>(copySize), // blockLen
            static_cast<uint32_t>(0),        // srcStride
            static_cast<uint32_t>(0),        // dstStride
            0                                // rsv
        };
        DataCopyPad(tGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

    template <typename T>
    __aicore__ inline void CopyInRPattern(
        TQue<QuePosition::VECIN, 1>& inQueue, GlobalTensor<T>& tGm, int64_t offset, uint32_t len)
    {
        LocalTensor<T> localValue = inQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),               // blockCount
            static_cast<uint32_t>(len * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),               // srcStride
            static_cast<uint32_t>(0),               // dstStride
            0                                       // rsv
        };
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0.0)};
        DataCopyPad(localValue, tGm[offset], copyParams, padParams);
        inQueue.EnQue(localValue);
    }

    __aicore__ inline void CopyInQuant(int64_t gmOffset, int64_t blockLen)
    {
        CopyInRPattern<T_SCALES>(inQueueScales1_, scales1Gm_, gmOffset, blockLen);
        if (isHasScale2_) {
            CopyInRPattern<T_SCALES>(inQueueScales2_, scales2Gm_, gmOffset, blockLen);
        }
        if (isHasZeroPoint1_) {
            CopyInRPattern<T_ZEROPOINTS>(inQueueZeroPoints1_, zeroPoints1Gm_, gmOffset, blockLen);
        }
        if (isHasZeroPoint2_) {
            CopyInRPattern<T_ZEROPOINTS>(inQueueZeroPoints2_, zeroPoints2Gm_, gmOffset, blockLen);
        }
    }

    __aicore__ inline void FoldBlockVF(
        LocalTensor<T_X>& xLocal, LocalTensor<T_X>& xFoldLocal, LocalTensor<float> xFp32Tmp, uint32_t tailCount,
        uint32_t count)
    {
        __local_mem__ T_X* xInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* xFp32TmpBuf = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();
        __local_mem__ T_X* xFoldInUb = (__local_mem__ T_X*)xFoldLocal.GetPhyAddr();

        uint16_t loops = (count + VL_FP32 - 1) / VL_FP32;
        uint16_t tailLoops = (tailCount + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, xFoldReg, sum;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sregTail = tailCount;
            for (uint16_t i = 0; i < tailLoops; ++i) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = i * VL_FP32;
                LoadTensorForDtypeTIn<T_X>(xInUb, xReg, pregFull, offset);
                LoadTensorForDtypeTIn<T_X>(xFoldInUb, xFoldReg, pregLoop, offset);
                Mul(xReg, xReg, xReg, pregFull);
                Mul(xFoldReg, xFoldReg, xFoldReg, pregLoop);
                Add(sum, xReg, xFoldReg, pregLoop);
                Select(sum, sum, xReg, pregLoop);
                DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32TmpBuf + offset, sum, pregFull);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(loops - tailLoops); ++i) {
                uint32_t offset = (i + tailLoops) * VL_FP32;
                LoadTensorForDtypeTIn<T_X>(xInUb, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32TmpBuf + offset, xReg, pregFull);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal, uint32_t reduceNum)
    {
        LocalTensor<float> binaryAddBuffTmp = binaryAddBuf_.Get<float>();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        if (reduceNum <= VL_FP32) {
            CalculateSquareReduceSumLessThanVL(xFp32Tmp, xReduceUb, reduceNum);
        } else if (reduceNum <= VL_FP32 + VL_FP32) {
            CalculateSquareReduceSumLessThanTwoVL(xFp32Tmp, xReduceUb, reduceNum);
        } else if (reduceNum <= VL_FP32 * VL_FP32 * NUM_TWO) {
            CalculateSquareReduceSumCommon<NUM_ONE>(xFp32Tmp, xReduceUb, tmpUb, reduceNum);
        } else {
            CalculateSquareReduceSumCommon<NUM_TWO>(xFp32Tmp, xReduceUb, tmpUb, reduceNum);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint32_t reduceNum)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(onesReg, float(1.0), pregOne);

            LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregLoop, 0);
            ReduceSum(vMean, x, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint32_t reduceNum)
    {
        uint32_t tailLen = reduceNum - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            Duplicate(onesReg, float(1.0), pregOne);

            LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, 0);
            LoadTensorForDtypeTIn<float>(xFp32Tmp + VL_FP32, xFold, pregTail, 0);
            ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
            Add(sumReg, x, xFold, pregFull);
            ReduceSum(vMean, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, __local_mem__ float* tmpUb, uint32_t reduceNum)
    {
        uint32_t binaryAddQuotient = binAddQuotient_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_FP32 - 1) / VL_FP32;
        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_FP32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_FP32 - 1) / VL_FP32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                uint32_t offset = r * VL_FP32;
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, offset);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddQuotient, xFold, pregFull, offset);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpUb + static_cast<uint32_t>(r), vMean, pregOne);
            }
            uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_FP32;
            for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop);
                 r++) {
                uint16_t offset = 0;
                pregLoop = UpdateMask<float>(sregRemainder);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_FP32, x, pregFull, 0);
                LoadTensorForDtypeTIn<float>(
                    xFp32Tmp + binaryAddRemainderFloorLoop * VL_FP32 + binaryAddQuotient, xFold, pregLoop, 0);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpUb + static_cast<uint32_t>(binaryAddRemainderFloorLoop), vMean, pregOne);
            }
            for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop); r++) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderCeilLoop * VL_FP32, x, pregFull, 0);
                ReduceSum(vMean, x, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpUb + static_cast<uint32_t>(binaryAddRemainderCeilLoop + r), vMean, pregOne);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                DataCopy(x, tmpUb);
                ReduceSum(vMean, x, pregLast);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_FP32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                DataCopy(x, tmpUb);
                DataCopy(xFold, tmpUb + VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateRstd(
        LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal, uint32_t curRows, float avgFactor,
        float epsilon)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        static constexpr float SCALAR1 = -0.5;
        static constexpr float SCALAR2 = 1.5;
        static constexpr float SCALAR3 = 0.5;
        static constexpr float SCALAR0 = -99.99;

        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>((curRows + VL_FP32 - 1) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> one;
            RegTensor<float> scalar1;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg = static_cast<uint32_t>(curRows);
            for (uint16_t i = 0; i < loopRows; ++i) {
                pregLoop = UpdateMask<float>(sreg);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(one, float(1.0), pregLoop);
                Duplicate(scalar1, SCALAR3, pregLoop);
                Duplicate(t1, SCALAR2, pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                // rstd
                DataCopy(var, xReduceUb + i * VL_FP32);
                Muls(var, var, avgFactor, pregLoop);
                Adds(var, var, epsilon, pregLoop);
                Maxs(var, var, SCALAR0, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, SCALAR1, pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdInUb + i * VL_FP32, rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void UpdateCache(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheId,
        const int64_t stride, const int64_t count)
    {
        uint16_t outerLoopTimes =
            ops::CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
        uint16_t innerLoopTimes = cacheId;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = stride;

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    AscendC::MicroAPI::DataCopy(
                        bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                AscendC::MicroAPI::DataCopy((__local_mem__ float*)cache + i * outerLoopStride, aReg, pMask);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> xGm_, gammaGm_, betaGm_;
    GlobalTensor<T_SCALES> scales1Gm_, scales2Gm_;
    GlobalTensor<T_ZEROPOINTS> zeroPoints1Gm_, zeroPoints2Gm_;
    GlobalTensor<yCopyDtype> y1Gm_, y2Gm_;

    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX_, inQueueXFold_, inQueueGamma_, inQueueBeta_;
    TQue<QuePosition::VECIN, 1> inQueueScales1_, inQueueScales2_;
    TQue<QuePosition::VECIN, 1> inQueueZeroPoints1_, inQueueZeroPoints2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_, outQueueY2_;

    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> binaryAddBuf_; // 整块 ub 二分累加
    TBuf<TPosition::VECCALC> cacheBuf_; // ub间 二分累加 需要的 cache buff, 最大值为 32B * log2(powerSplit)
    TBuf<TPosition::VECCALC> xFp32TmpBuf_; // 长度 baseN_ , float

    // Tiling data
    int64_t numN_{0};
    int64_t numM_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mPerCore_{0};
    int64_t mLastCore_{0};
    int64_t nUbLoops_{0};
    int64_t binAddQuotient_{0};
    int64_t powerSplit_;
    int64_t mainFoldCount_;
    int64_t foldTail_;
    uint32_t resultCacheID_{0};
    uint32_t option_mask_{0};
    uint32_t div_mode_;
    uint32_t dst_type_;
    float epsilon_{0};
    float avgFactor_{0};

    // Cal params
    int64_t mCurCore_;
    bool isHasScale2_;
    bool isHasZeroPoint1_;
    bool isHasZeroPoint2_;
    bool isHasY2_;
    bool isHasBeta_;
    bool isNeedBrc_;
};
} // namespace RmsNormQuantV2
#endif // RMS_NORM_QUANT_V2_REBASE_REDUCE_H_
