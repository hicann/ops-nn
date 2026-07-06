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
 * \file rms_norm_grad_regbase_dx_split_d.h
 * \brief RmsNormGrad Regbase DX Split D kernel File
 */

#ifndef RMS_NORM_GRAD_REGBASE_DX_SPLIT_D_H
#define RMS_NORM_GRAD_REGBASE_DX_SPLIT_D_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "rms_norm_grad_quant_common.h"

namespace RmsNormGradQuant {
using namespace AscendC;
template <typename T_DY, typename T_X, typename T_GAMMA, typename T_DX, typename T_DGAMMA, typename T_SCALES_X,
          typename T_OFFSET_X, bool HAS_OFFSET_X, bool DIV_MODE>
class RegbaseDxSplitD {
public:
    __aicore__ inline RegbaseDxSplitD(TPipe* pipe, const RmsNormGradQuantRegbaseDxTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* rstd, __gm__ uint8_t* gamma,
                                __gm__ uint8_t* scales_x, __gm__ uint8_t* offset_x, __gm__ uint8_t* dx,
                                __gm__ uint8_t* dgamma)
    {
#if (__NPU_ARCH__ == 3510)
        if constexpr (IsSameType<T_DX, hifloat8_t>::value) {
            AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
        }
#endif
        usedCoreNum_ = tiling_->usedCoreNumDx;
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        rows_ = tiling_->rows;
        cols_ = tiling_->cols;
        blockFactor_ = tiling_->blockFactorDx; // ceilDiv(rows_, usedCoreNum)
        ubFactorD_ = UB_FACTOR_DX_SPLIT_D;     // 固定值
        bodyPart_ = tiling_->bodyPart;         // 小于cols的最大二次幂
        avgFactor1_ = 1.0f / cols_;
        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + coreIdx * blockFactor_ * cols_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + coreIdx * blockFactor_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        dxGm_.SetGlobalBuffer((__gm__ T_DX*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueRstd_, DB_NUM, V_LENGTH * sizeof(float));
        Ppipe_->InitBuffer(inQueueGamma_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBuf_, DB_NUM * ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(level0Buf_, ONCE_VECTOR_SIZE * sizeof(float));
        Ppipe_->InitBuffer(level1Buf_, ONCE_VECTOR_SIZE * sizeof(float));
        Ppipe_->InitBuffer(level2Buf_, ONCE_VECTOR_SIZE * sizeof(float));
        Ppipe_->InitBuffer(tmpSumBuf_, V_LENGTH * sizeof(float));
        Ppipe_->InitBuffer(workBuf_, ONCE_VECTOR_SIZE * sizeof(float));
        scalesXGm_.SetGlobalBuffer((__gm__ T_SCALES_X*)scales_x);
        Ppipe_->InitBuffer(scalesXBuf_, sizeof(T_SCALES_X));
        if constexpr (HAS_OFFSET_X) {
            offsetXGm_.SetGlobalBuffer((__gm__ T_OFFSET_X*)offset_x);
            Ppipe_->InitBuffer(offsetXBuf_, sizeof(T_OFFSET_X));
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        // copyInScalesX
        CopyInScalesX();
        if constexpr (HAS_OFFSET_X) {
            CopyInOffsetX();
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx++) {
            SubProcess(rowIdx);
        }
    }

    __aicore__ inline void SubProcess(int64_t rowIdx)
    {
        CopyInRstd(rowIdx, 1);
        LocalTensor<float> rstdLocal = inQueueRstd_.DeQue<float>();
        FormerProcess(rstdLocal, rowIdx);
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();
        Muls(tmpSumLocal, tmpSumLocal, avgFactor1_, 1);
        LatterProcess(rstdLocal, rowIdx);
        inQueueRstd_.FreeTensor(rstdLocal);
    }

    __aicore__ inline void FormerProcess(LocalTensor<float>& rstdLocal, int64_t rowIdx)
    {
        uint32_t level0Offset = 0;
        uint32_t level1Offset = 0;
        uint32_t level2Offset = 0;
        InitLevelLocal();
        for (int64_t colIdx = 0; colIdx < bodyPart_; colIdx += ubFactorD_) {
            CopyInGamma(colIdx, ubFactorD_);
            CopyInDy(rowIdx, colIdx, ubFactorD_);
            CopyInX(rowIdx, colIdx, ubFactorD_);
            ComputeMul<true>(rstdLocal, ubFactorD_);
            int64_t tailCount = 0;
            if (bodyPart_ + colIdx < cols_) {
                int64_t remainCount = cols_ - bodyPart_ - colIdx; // must > 0
                tailCount = Min(remainCount, ubFactorD_);
                CopyInGamma(bodyPart_ + colIdx, tailCount);
                CopyInDy(rowIdx, bodyPart_ + colIdx, tailCount);
                CopyInX(rowIdx, bodyPart_ + colIdx, tailCount);
                ComputeMul<false>(rstdLocal, tailCount);
            }
            int64_t reduceCount = ubFactorD_ + tailCount; // [ubFactorD, 2*ubFactorD_]
            ComputeIntoMultiLevel(reduceCount, level0Offset, level1Offset, level2Offset);
        }
        FinalLevelReduce(level0Offset, level1Offset);
    }

    __aicore__ inline void InitLevelLocal()
    {
        LocalTensor<float> level0Local = level0Buf_.Get<float>();
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();

        Duplicate(level0Local, 0.0f, ONCE_VECTOR_SIZE);
        Duplicate(level1Local, 0.0f, ONCE_VECTOR_SIZE);
        Duplicate(level2Local, 0.0f, ONCE_VECTOR_SIZE);
    }

    __aicore__ inline void ComputeIntoMultiLevel(int64_t count, uint32_t& level0Offset, uint32_t& level1Offset,
                                                 uint32_t& level2Offset)
    {
        LocalTensor<float> level0Local = level0Buf_.Get<float>();
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        WholeReduceSum(level0Local, reduceLocal, count, level0Offset);
        level0Offset++;
        ComputeMultiLevelReduce(level0Local, level1Local, level2Local, level0Offset, level1Offset, level2Offset);
    }

    __aicore__ inline void FinalLevelReduce(uint32_t& level0Offset, uint32_t& level1Offset)
    {
        LocalTensor<float> level0Local = level0Buf_.Get<float>();
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();
        ComputeMultiLevelMean(tmpSumLocal, 0, level0Local, level1Local, level2Local, level0Offset, level1Offset);
    }

    __aicore__ inline void CopyInRstd(int64_t rowIdx, int64_t count)
    {
        LocalTensor<float> rstdLocal = inQueueRstd_.AllocTensor<float>();
        DataCopyExtParams copyParams{
            1,                                            // blockCount
            static_cast<uint32_t>(count * sizeof(float)), // blockLen
            0,                                            // srcStride
            0,                                            // dstStride
            0                                             // rsv
        };
        DataCopyPad(rstdLocal, rstdGm_[rowIdx], copyParams, {true, 0, 0, 0});
        inQueueRstd_.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyInGamma(int64_t colIdx, int64_t count)
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma_.AllocTensor<T_GAMMA>();
        DataCopyExtParams copyParams{
            1,                                              // blockCount
            static_cast<uint32_t>(count * sizeof(T_GAMMA)), // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };
        DataCopyPad(gammaLocal, gammaGm_[colIdx], copyParams, {true, 0, 0, 0});
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInDy(int64_t rowIdx, int64_t colIdx, int64_t count)
    {
        LocalTensor<T_DY> dyLocal = inQueueDy_.AllocTensor<T_DY>();
        DataCopyExtParams copyParams{
            1,                                           // blockCount
            static_cast<uint32_t>(count * sizeof(T_DY)), // blockLen
            0,                                           // srcStride
            0,                                           // dstStride
            0                                            // rsv
        };
        DataCopyPad(dyLocal, dyGm_[rowIdx * cols_ + colIdx], copyParams, {true, 0, 0, 0});
        inQueueDy_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInX(int64_t rowIdx, int64_t colIdx, int64_t count)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        DataCopyExtParams copyParams{
            1,                                          // blockCount
            static_cast<uint32_t>(count * sizeof(T_X)), // blockLen
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        DataCopyPad(xLocal, xGm_[rowIdx * cols_ + colIdx], copyParams, {true, 0, 0, 0});
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInScalesX()
    {
        LocalTensor<T_SCALES_X> scalesXLocal = scalesXBuf_.Get<T_SCALES_X>();
        DataCopyExtParams copyParams{
            1,                                             // blockCount
            static_cast<uint32_t>(1 * sizeof(T_SCALES_X)), // blockLen
            0,                                             // srcStride
            0,                                             // dstStride
            0                                              // rsv
        };

        DataCopyPad(scalesXLocal, scalesXGm_, copyParams, {true, 0, 0, 0});
    }

    __aicore__ inline void CopyInOffsetX()
    {
        LocalTensor<T_OFFSET_X> offsetXLocal = offsetXBuf_.Get<T_OFFSET_X>();
        DataCopyExtParams copyParams{
            1,                                             // blockCount
            static_cast<uint32_t>(1 * sizeof(T_OFFSET_X)), // blockLen
            0,                                             // srcStride
            0,                                             // dstStride
            0                                              // rsv
        };

        DataCopyPad(offsetXLocal, offsetXGm_, copyParams, {true, 0, 0, 0});
    }

    template <typename T_IN>
    __aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg)
    {
        if constexpr (IsSameType<T_IN, float>::value) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst, src);
        } else if constexpr (IsSameType<T_IN, int32_t>::value) {
            RegTensor<T_IN> xIn;
            DataCopy<int32_t, LoadDist::DIST_BRC_B32>(xIn, src);
            Cast<float, T_IN, castTraitInt322Fp32>(dst, xIn, preg);
        } else {
            RegTensor<T_IN> xIn;
            DataCopy<T_IN, LoadDist::DIST_BRC_B16>(xIn, src);
            Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
        }
    }

    template <bool IsBody>
    __aicore__ inline void ComputeMul(LocalTensor<float>& rstdLocal, int64_t count)
    {
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        LocalTensor<float> gammaLocal = inQueueGamma_.DeQue<float>();
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();

        uint32_t sreg = count;
        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = DivCeil(count, oneRepeat);
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_DY* dyAddr = (__ubuf__ T_DY*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* reduceAddr = (__ubuf__ float*)reduceLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, mulReg0, mulReg2, mulReg3;
            MaskReg maskReg = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            for (uint16_t i = 0; i < repeatCount; i++) {
                LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                Mul(mulReg2, dyReg, gammaReg, maskReg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(mulReg0, xReg, rstdReg, maskReg);
                Mul(mulReg3, mulReg2, mulReg0, maskReg);
                if constexpr (IsBody) {
                    DataCopy(reduceAddr + static_cast<uint32_t>(i * oneRepeat), mulReg3, maskReg);
                } else {
                    DataCopy(reduceAddr + static_cast<uint32_t>(ubFactorD_ + i * oneRepeat), mulReg3,
                             maskReg); // 注意补零
                }
            }
        }

        inQueueGamma_.FreeTensor(gammaLocal);
        inQueueDy_.FreeTensor(dyLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void WholeReduceSum(LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, int64_t count,
                                          int32_t dstOffset)
    {
        // 对齐到512BYTE, reduce需要
        int64_t countBlockAlign = AlignUp(count, FLOAT_NUM_BLOCK); // 搬入已对齐
        int64_t count2VLAlign = AlignUp(count, FLOAT_NUM_2VL);
        if (count2VLAlign - countBlockAlign > 0) {
            Duplicate(srcLocal[countBlockAlign], 0.0f, count2VLAlign - countBlockAlign);
        }
        int64_t power = count2VLAlign < NUM_TWO * ubFactorD_ ?
                            ubFactorD_ :
                            NUM_TWO * ubFactorD_; // 等于2*UbFactorD_时设为相同大小，否则为其一半
        LocalTensor<float> workLocal = workBuf_.Get<float>();
        ReduceSumImpl(dstLocal, srcLocal, workLocal, dstOffset, count2VLAlign, power);
    }

    __aicore__ inline void LatterProcess(LocalTensor<float>& rstdLocal, int64_t rowIdx)
    {
        for (int64_t colIdx = 0; colIdx < cols_; colIdx += ubFactorD_) {
            int64_t remainCount = cols_ - colIdx;
            int64_t calcCount = Min(remainCount, ubFactorD_);
            CopyInGamma(colIdx, calcCount);
            CopyInDy(rowIdx, colIdx, calcCount);
            CopyInX(rowIdx, colIdx, calcCount);
            ComputeLatter(rstdLocal, calcCount);
            CopyOutDx(rowIdx, colIdx, calcCount);
        }
    }

    __aicore__ inline void ComputeLatter(LocalTensor<float>& rstdLocal, int64_t count)
    {
        LocalTensor<float> gammaLocal = inQueueGamma_.DeQue<float>();
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        LocalTensor<T_DX> dxLocal = outQueueDx_.AllocTensor<T_DX>();
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();
        LocalTensor<T_SCALES_X> scalesXLocal;
        LocalTensor<T_OFFSET_X> offsetXLocal;

        uint32_t sreg = count;
        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = DivCeil(count, oneRepeat); // 可能会报错
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_DY* dyAddr = (__ubuf__ T_DY*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* meanAddr = (__ubuf__ float*)tmpSumLocal.GetPhyAddr();
        __local_mem__ T_DX* dxAddr = (__ubuf__ T_DX*)dxLocal.GetPhyAddr();
        __local_mem__ T_SCALES_X* scalesXAddr;
        __local_mem__ T_OFFSET_X* offsetXAddr;

        scalesXLocal = scalesXBuf_.Get<T_SCALES_X>();
        scalesXAddr = (__ubuf__ T_SCALES_X*)scalesXLocal.GetPhyAddr();
        if constexpr (HAS_OFFSET_X) {
            offsetXLocal = offsetXBuf_.Get<T_OFFSET_X>();
            offsetXAddr = (__ubuf__ T_OFFSET_X*)offsetXLocal.GetPhyAddr();
        }

        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, meanReg, dxReg, mulReg0, mulReg2, mulReg4, subReg;
            RegTensor<float> scalesXReg, scalesXResultReg, offsetXReg;
            MaskReg maskReg;
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            for (uint16_t i = 0; i < repeatCount; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                Mul(mulReg2, dyReg, gammaReg, maskReg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(mulReg0, xReg, rstdReg, maskReg);
                Mul(mulReg4, mulReg0, meanReg, maskReg);
                Sub(subReg, mulReg2, mulReg4, maskReg);
                Mul(dxReg, subReg, rstdReg, maskReg);
                // cal quant
                LoadTensorForDtypeTIn(scalesXAddr, scalesXReg, maskReg);
                if constexpr (DIV_MODE) {
                    Div(scalesXResultReg, dxReg, scalesXReg, maskReg);
                } else {
                    Mul(scalesXResultReg, dxReg, scalesXReg, maskReg);
                }
                if constexpr (HAS_OFFSET_X) {
                    LoadTensorForDtypeTIn(offsetXAddr, offsetXReg, maskReg);
                    Add(scalesXResultReg, scalesXResultReg, offsetXReg, maskReg);
                }
                if constexpr (IsSameType<T_DX, hifloat8_t>::value) {
                    RegTensor<T_DX> dxRegHif8;
                    Cast<T_DX, float, castTraitFp322Hifp8>(dxRegHif8, scalesXResultReg, maskReg);
                    DataCopy<T_DX, StoreDist::DIST_PACK4_B32>(dxAddr + static_cast<uint32_t>(i * oneRepeat), dxRegHif8,
                                                              maskReg);
                } else if constexpr (IsSameType<T_DX, int8_t>::value) {
                    RegTensor<T_DX> dxRegInt8;
                    RegTensor<half> dxRegFp16;
                    RegTensor<int32_t> dxRegInt32;
                    Cast<int32_t, float, castTraitFp322Int32>(dxRegInt32, scalesXResultReg, maskReg);
                    Cast<float, int32_t, castTraitInt322Fp32>(scalesXResultReg, dxRegInt32, maskReg);
                    Cast<half, float, castTraitFp322Fp16>(dxRegFp16, scalesXResultReg, maskReg);
                    Cast<T_DX, half, castTraitFp162Int8>(dxRegInt8, dxRegFp16, maskReg);
                    DataCopy<T_DX, StoreDist::DIST_PACK4_B32>(dxAddr + static_cast<uint32_t>(i * oneRepeat), dxRegInt8,
                                                              maskReg);
                }
            }
        }

        inQueueGamma_.FreeTensor(gammaLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t colIdx, int64_t count)
    {
        LocalTensor<T_DX> dxLocal = outQueueDx_.DeQue<T_DX>();
        DataCopyExtParams copyParams{
            1,                                           // blockCount
            static_cast<uint32_t>(count * sizeof(T_DX)), // blockLen
            0,                                           // srcStride
            0,                                           // dstStride
            0                                            // rsv
        };
        DataCopyPad(dxGm_[rowIdx * cols_ + colIdx], dxLocal, copyParams);
        outQueueDx_.FreeTensor(dxLocal);
    }

private:
    TPipe* Ppipe_;
    const RmsNormGradQuantRegbaseDxTilingData* tiling_;
    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<T_DX> dxGm_;
    GlobalTensor<T_SCALES_X> scalesXGm_;
    GlobalTensor<T_OFFSET_X> offsetXGm_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueRstd_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueGamma_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> level0Buf_;
    TBuf<TPosition::VECCALC> level1Buf_;
    TBuf<TPosition::VECCALC> level2Buf_;
    TBuf<TPosition::VECCALC> tmpSumBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> scalesXBuf_;
    TBuf<TPosition::VECCALC> offsetXBuf_;
    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t blockFactor_;
    int64_t ubFactorD_;
    int64_t bodyPart_;
    float avgFactor1_;
};
} // namespace RmsNormGradQuant
#endif // RMS_NORM_GRAD_REGBASE_DX_SPLIT_D_H