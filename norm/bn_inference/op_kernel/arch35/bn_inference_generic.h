/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_INFERENCE_GENERIC_H
#define BN_INFERENCE_GENERIC_H

#include "bn_inference_common.h"
#include "bn_inference_tiling_data.h"

namespace BNInferenceOps {
template <typename T_X, typename T_MEAN, typename T_VARIANCE, typename T_MOMENTUM, typename T_SCALE, typename T_OFFSET,
          bool HAS_SCALE, bool HAS_OFFSET, bool CHANNEL_LAST, bool PRE_FOLDED>
class BNInferenceGeneric {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mean, GM_ADDR variance, GM_ADDR momentum, GM_ADDR scale,
                                GM_ADDR offset, GM_ADDR y, const BNInferenceTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        meanGm_.SetGlobalBuffer((__gm__ T_MEAN*)mean);
        varianceGm_.SetGlobalBuffer((__gm__ T_VARIANCE*)variance);
        if constexpr (!PRE_FOLDED && !HAS_SCALE && !HAS_OFFSET) {
            momentumGm_.SetGlobalBuffer((__gm__ T_MOMENTUM*)momentum, 1);
        }
        if constexpr (HAS_SCALE) {
            scaleGm_.SetGlobalBuffer((__gm__ T_SCALE*)scale);
        }
        if constexpr (HAS_OFFSET) {
            offsetGm_.SetGlobalBuffer((__gm__ T_OFFSET*)offset);
        }
        yGm_.SetGlobalBuffer((__gm__ T_X*)y);

        int64_t maxXElements = CHANNEL_LAST ? tilingData_->tileRows * tilingData_->tileElements :
                                              tilingData_->tileElements;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, AlignUpBlock(maxXElements * sizeof(T_X)));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, AlignUpBlock(maxXElements * sizeof(T_X)));
        pipe_->InitBuffer(meanQueue_, BUFFER_NUM, AlignUpBlock(tilingData_->paramTileLen * sizeof(T_MEAN)));
        pipe_->InitBuffer(varianceQueue_, BUFFER_NUM, AlignUpBlock(tilingData_->paramTileLen * sizeof(T_VARIANCE)));
        if constexpr (!PRE_FOLDED && !HAS_SCALE && !HAS_OFFSET) {
            pipe_->InitBuffer(momentumBuffer_, UB_BLOCK_BYTES);
        }
        if constexpr (HAS_SCALE) {
            pipe_->InitBuffer(scaleQueue_, BUFFER_NUM, AlignUpBlock(tilingData_->paramTileLen * sizeof(T_SCALE)));
            pipe_->InitBuffer(scaleCache_, AlignUpBlock(tilingData_->paramCacheLen * sizeof(float)));
        }
        if constexpr (HAS_OFFSET) {
            pipe_->InitBuffer(offsetQueue_, BUFFER_NUM, AlignUpBlock(tilingData_->paramTileLen * sizeof(T_OFFSET)));
            pipe_->InitBuffer(offsetCache_, AlignUpBlock(tilingData_->paramCacheLen * sizeof(float)));
        }
        pipe_->InitBuffer(meanCache_, AlignUpBlock(tilingData_->paramCacheLen * sizeof(float)));
        pipe_->InitBuffer(rstdCache_, AlignUpBlock(tilingData_->paramCacheLen * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        if constexpr (!PRE_FOLDED && !HAS_SCALE && !HAS_OFFSET) {
            LoadMomentumFactor();
        }
        const int64_t blockIdx = GetBlockIdx();
        const int64_t tileCount = tilingData_->baseTilesPerCore + (blockIdx < tilingData_->extraCoreCount ? 1 : 0);
        const int64_t beginTile = blockIdx * tilingData_->baseTilesPerCore +
                                  (blockIdx < tilingData_->extraCoreCount ? blockIdx : tilingData_->extraCoreCount);
        int64_t cachedCBegin = -1;
        int64_t cachedCLen = -1;
        for (int64_t localTile = 0; localTile < tileCount; ++localTile) {
            const int64_t tileId = beginTile + localTile;
            if constexpr (CHANNEL_LAST) {
                ProcessChannelLastTile(tileId, cachedCBegin, cachedCLen);
            } else {
                ProcessChannelFirstTile(tileId, cachedCBegin, cachedCLen);
            }
        }
    }

private:
    __aicore__ inline void LoadMomentumFactor()
    {
        LocalTensor<T_MOMENTUM> momentum = momentumBuffer_.Get<T_MOMENTUM>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T_MOMENTUM)), 0, 0, 0};
        DataCopyPadExtParams<T_MOMENTUM> pad{false, 0, 0, 0};
        DataCopyPad(momentum, momentumGm_, params, pad);
        event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        float momentumValue = 0.0f;
        if constexpr (IsSameType<T_MOMENTUM, bfloat16_t>::value) {
            momentumValue = AscendC::ToFloat(momentum.GetValue(0));
        } else {
            momentumValue = static_cast<float>(momentum.GetValue(0));
        }
        factor_ = momentumValue == 0.0f ? 0.0f : 1.0f / momentumValue;
        if constexpr (IsSameType<T_MOMENTUM, half>::value) {
            factor_ = static_cast<float>(static_cast<half>(factor_));
        }
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> StageParameter(TQue<QuePosition::VECIN, 1>& queue, const GlobalTensor<T>& gm,
                                                    int64_t offset, int64_t count)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], params, pad);
        queue.EnQue(local);
        return queue.template DeQue<T>();
    }

    __aicore__ inline void PrepareParameters(int64_t cBegin, int64_t cLen)
    {
        LocalTensor<T_MEAN> mean = StageParameter(meanQueue_, meanGm_, cBegin, cLen);
        LocalTensor<T_VARIANCE> variance = StageParameter(varianceQueue_, varianceGm_, cBegin, cLen);
        LocalTensor<T_SCALE> scale;
        LocalTensor<T_OFFSET> offset;
        __ubuf__ T_SCALE* scaleAddr = nullptr;
        __ubuf__ T_OFFSET* offsetAddr = nullptr;
        if constexpr (HAS_SCALE) {
            scale = StageParameter(scaleQueue_, scaleGm_, cBegin, cLen);
            scaleAddr = (__ubuf__ T_SCALE*)scale.GetPhyAddr();
        }
        if constexpr (HAS_OFFSET) {
            offset = StageParameter(offsetQueue_, offsetGm_, cBegin, cLen);
            offsetAddr = (__ubuf__ T_OFFSET*)offset.GetPhyAddr();
        }

        VFPrepareParameters((__ubuf__ T_MEAN*)mean.GetPhyAddr(), (__ubuf__ T_VARIANCE*)variance.GetPhyAddr(), scaleAddr,
                            offsetAddr, (__ubuf__ float*)meanCache_.Get<float>().GetPhyAddr(),
                            (__ubuf__ float*)rstdCache_.Get<float>().GetPhyAddr(),
                            HAS_SCALE ? (__ubuf__ float*)scaleCache_.Get<float>().GetPhyAddr() : nullptr,
                            HAS_OFFSET ? (__ubuf__ float*)offsetCache_.Get<float>().GetPhyAddr() : nullptr,
                            static_cast<uint16_t>(cLen));

        meanQueue_.FreeTensor(mean);
        varianceQueue_.FreeTensor(variance);
        if constexpr (HAS_SCALE) {
            scaleQueue_.FreeTensor(scale);
        }
        if constexpr (HAS_OFFSET) {
            offsetQueue_.FreeTensor(offset);
        }
    }

    __aicore__ inline void VFPrepareParameters(__ubuf__ T_MEAN* mean, __ubuf__ T_VARIANCE* variance,
                                               __ubuf__ T_SCALE* scale, __ubuf__ T_OFFSET* offset,
                                               __ubuf__ float* meanCache, __ubuf__ float* rstdCache,
                                               __ubuf__ float* scaleCache, __ubuf__ float* offsetCache, uint16_t count)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> meanReg;
            RegTensor<float> varianceReg;
            RegTensor<float> rstdReg;
            RegTensor<float> scaleReg;
            RegTensor<float> offsetReg;
            RegTensor<float> alphaReg;
            RegTensor<float> betaReg;
            const uint16_t loopCount = (count + VL_FP32 - 1) / VL_FP32;
            for (uint16_t i = 0; i < loopCount; ++i) {
                const uint16_t processed = i * VL_FP32;
                const uint32_t active = count - processed > VL_FP32 ? VL_FP32 : count - processed;
                uint32_t maskCount = active;
                MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                LoadUnalignedOnce(mean + processed, meanReg, mask, active);
                LoadUnalignedOnce(variance + processed, varianceReg, mask, active);
                if constexpr (HAS_SCALE) {
                    LoadUnalignedOnce(scale + processed, scaleReg, mask, active);
                }
                if constexpr (HAS_OFFSET) {
                    LoadUnalignedOnce(offset + processed, offsetReg, mask, active);
                }
                if constexpr (PRE_FOLDED) {
                    StoreFromFp32(meanCache, meanReg, mask, processed);
                    StoreFromFp32(rstdCache, varianceReg, mask, processed);
                } else if constexpr (!HAS_SCALE && !HAS_OFFSET) {
                    FoldWithoutAffine<T_MEAN, T_VARIANCE>(meanReg, varianceReg, alphaReg, betaReg, mask,
                                                          tilingData_->epsilon, factor_);
                    StoreFromFp32(meanCache, alphaReg, mask, processed);
                    StoreFromFp32(rstdCache, betaReg, mask, processed);
                } else if constexpr (HAS_SCALE && HAS_OFFSET) {
                    FoldWithAffine<T_MEAN, T_VARIANCE>(meanReg, varianceReg, scaleReg, offsetReg, alphaReg, betaReg,
                                                       mask, tilingData_->epsilon);
                    StoreFromFp32(meanCache, alphaReg, mask, processed);
                    StoreFromFp32(rstdCache, betaReg, mask, processed);
                } else {
                    ComputeRstdExact(varianceReg, rstdReg, mask, tilingData_->epsilon);
                    StoreFromFp32(meanCache, meanReg, mask, processed);
                    StoreFromFp32(rstdCache, rstdReg, mask, processed);
                }
                if constexpr (HAS_SCALE) {
                    StoreFromFp32(scaleCache, scaleReg, mask, processed);
                }
                if constexpr (HAS_OFFSET) {
                    StoreFromFp32(offsetCache, offsetReg, mask, processed);
                }
            }
        }
    }

    __aicore__ inline void EnsureParameters(int64_t cBegin, int64_t cLen, int64_t& cachedCBegin, int64_t& cachedCLen)
    {
        if (cBegin != cachedCBegin || cLen != cachedCLen) {
            PrepareParameters(cBegin, cLen);
            cachedCBegin = cBegin;
            cachedCLen = cLen;
        }
    }

    __aicore__ inline void CopyInContiguous(int64_t gmOffset, int64_t count)
    {
        LocalTensor<T_X> x = xQueue_.AllocTensor<T_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T_X)), 0, 0, 0};
        DataCopyPadExtParams<T_X> pad{false, 0, 0, 0};
        DataCopyPad<T_X, PaddingMode::Compact>(x, xGm_[gmOffset], params, pad);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void CopyOutContiguous(int64_t gmOffset, int64_t count)
    {
        LocalTensor<T_X> y = yQueue_.DeQue<T_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T_X)), 0, 0, 0};
        DataCopyPad<T_X, PaddingMode::Compact>(yGm_[gmOffset], y, params);
        yQueue_.FreeTensor(y);
    }

    __aicore__ inline void CopyInChannelLast(int64_t gmOffset, int64_t rows, int64_t cLen)
    {
        LocalTensor<T_X> x = xQueue_.AllocTensor<T_X>();
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cLen * sizeof(T_X)),
                                 (tilingData_->c - cLen) * static_cast<int64_t>(sizeof(T_X)), 0, 0};
        DataCopyPadExtParams<T_X> pad{false, 0, 0, 0};
        DataCopyPad<T_X, PaddingMode::Compact>(x, xGm_[gmOffset], params, pad);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void CopyOutChannelLast(int64_t gmOffset, int64_t rows, int64_t cLen)
    {
        LocalTensor<T_X> y = yQueue_.DeQue<T_X>();
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cLen * sizeof(T_X)), 0,
                                 (tilingData_->c - cLen) * static_cast<int64_t>(sizeof(T_X)), 0};
        DataCopyPad<T_X, PaddingMode::Compact>(yGm_[gmOffset], y, params);
        yQueue_.FreeTensor(y);
    }

    __aicore__ inline void ComputeChannelFirst(uint16_t count)
    {
        LocalTensor<T_X> x = xQueue_.DeQue<T_X>();
        LocalTensor<T_X> y = yQueue_.AllocTensor<T_X>();
        __ubuf__ T_X* xAddr = (__ubuf__ T_X*)x.GetPhyAddr();
        __ubuf__ T_X* yAddr = (__ubuf__ T_X*)y.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)meanCache_.Get<float>().GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdCache_.Get<float>().GetPhyAddr();
        __ubuf__ float* scaleAddr = HAS_SCALE ? (__ubuf__ float*)scaleCache_.Get<float>().GetPhyAddr() : nullptr;
        __ubuf__ float* offsetAddr = HAS_OFFSET ? (__ubuf__ float*)offsetCache_.Get<float>().GetPhyAddr() : nullptr;
        VFComputeChannelFirst(xAddr, yAddr, meanAddr, rstdAddr, scaleAddr, offsetAddr, count);
        yQueue_.EnQue(y);
        xQueue_.FreeTensor(x);
    }

    __aicore__ inline void VFComputeChannelFirst(__ubuf__ T_X* x, __ubuf__ T_X* y, __ubuf__ float* mean,
                                                 __ubuf__ float* rstd, __ubuf__ float* scale, __ubuf__ float* offset,
                                                 uint16_t count)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            RegTensor<float> meanReg;
            RegTensor<float> rstdReg;
            RegTensor<float> scaleReg;
            RegTensor<float> offsetReg;
            MaskReg allMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(meanReg, mean);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstd);
            if constexpr (HAS_SCALE) {
                AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, scale);
            } else {
                AscendC::Reg::Duplicate(scaleReg, 1.0f, allMask);
            }
            if constexpr (HAS_OFFSET) {
                AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(offsetReg, offset);
            } else {
                AscendC::Reg::Duplicate(offsetReg, 0.0f, allMask);
            }
            AscendC::MicroAPI::UnalignReg xState;
            AscendC::MicroAPI::UnalignReg yState;
            __ubuf__ T_X* xCurrent = x;
            __ubuf__ T_X* yCurrent = y;
            AscendC::MicroAPI::DataCopyUnAlignPre(xState, xCurrent);
            const uint16_t loopCount = (count + VL_FP32 - 1) / VL_FP32;
            for (uint16_t i = 0; i < loopCount; ++i) {
                const uint16_t processed = i * VL_FP32;
                const uint32_t active = count - processed > VL_FP32 ? VL_FP32 : count - processed;
                uint32_t maskCount = active;
                MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                LoadUnalignedToFp32(xCurrent, xReg, xState, mask, active);
                if constexpr (PRE_FOLDED) {
                    ApplyPreFoldedAffine<T_X, HAS_SCALE, HAS_OFFSET>(xReg, meanReg, rstdReg, scaleReg, offsetReg, yReg,
                                                                     mask);
                } else if constexpr (HAS_SCALE == HAS_OFFSET) {
                    ApplyFoldedAffine<T_X>(xReg, meanReg, rstdReg, yReg, mask);
                } else {
                    NormalizeOneSided<HAS_SCALE, HAS_OFFSET>(xReg, meanReg, rstdReg, scaleReg, offsetReg, yReg, mask);
                }
                StoreUnalignedFromFp32(yCurrent, yReg, yState, mask, active);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(yCurrent, yState, 0);
        }
    }

    __aicore__ inline void ComputeChannelLast(uint16_t rows, uint16_t cLen)
    {
        LocalTensor<T_X> x = xQueue_.DeQue<T_X>();
        LocalTensor<T_X> y = yQueue_.AllocTensor<T_X>();
        VFComputeChannelLast((__ubuf__ T_X*)x.GetPhyAddr(), (__ubuf__ T_X*)y.GetPhyAddr(),
                             (__ubuf__ float*)meanCache_.Get<float>().GetPhyAddr(),
                             (__ubuf__ float*)rstdCache_.Get<float>().GetPhyAddr(),
                             HAS_SCALE ? (__ubuf__ float*)scaleCache_.Get<float>().GetPhyAddr() : nullptr,
                             HAS_OFFSET ? (__ubuf__ float*)offsetCache_.Get<float>().GetPhyAddr() : nullptr, rows,
                             cLen);
        yQueue_.EnQue(y);
        xQueue_.FreeTensor(x);
    }

    __aicore__ inline void VFComputeChannelLast(__ubuf__ T_X* x, __ubuf__ T_X* y, __ubuf__ float* mean,
                                                __ubuf__ float* rstd, __ubuf__ float* scale, __ubuf__ float* offset,
                                                uint16_t rows, uint16_t cLen)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            RegTensor<float> meanReg;
            RegTensor<float> rstdReg;
            RegTensor<float> scaleReg;
            RegTensor<float> offsetReg;
            for (uint16_t row = 0; row < rows; ++row) {
                __ubuf__ T_X* xCurrent = x + row * cLen;
                __ubuf__ T_X* yCurrent = y + row * cLen;
                AscendC::MicroAPI::UnalignReg xState;
                AscendC::MicroAPI::UnalignReg yState;
                AscendC::MicroAPI::DataCopyUnAlignPre(xState, xCurrent);
                const uint16_t loopCount = (cLen + VL_FP32 - 1) / VL_FP32;
                for (uint16_t i = 0; i < loopCount; ++i) {
                    const uint16_t processed = i * VL_FP32;
                    const uint32_t active = cLen - processed > VL_FP32 ? VL_FP32 : cLen - processed;
                    uint32_t maskCount = active;
                    MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                    LoadUnalignedToFp32(xCurrent, xReg, xState, mask, active);
                    LoadUnalignedOnce(mean + processed, meanReg, mask, active);
                    LoadUnalignedOnce(rstd + processed, rstdReg, mask, active);
                    if constexpr (HAS_SCALE) {
                        LoadUnalignedOnce(scale + processed, scaleReg, mask, active);
                    } else {
                        AscendC::Reg::Duplicate(scaleReg, 1.0f, mask);
                    }
                    if constexpr (HAS_OFFSET) {
                        LoadUnalignedOnce(offset + processed, offsetReg, mask, active);
                    } else {
                        AscendC::Reg::Duplicate(offsetReg, 0.0f, mask);
                    }
                    if constexpr (PRE_FOLDED) {
                        ApplyPreFoldedAffine<T_X, HAS_SCALE, HAS_OFFSET>(xReg, meanReg, rstdReg, scaleReg, offsetReg,
                                                                         yReg, mask);
                    } else if constexpr (HAS_SCALE == HAS_OFFSET) {
                        ApplyFoldedAffine<T_X>(xReg, meanReg, rstdReg, yReg, mask);
                    } else {
                        NormalizeOneSided<HAS_SCALE, HAS_OFFSET>(xReg, meanReg, rstdReg, scaleReg, offsetReg, yReg,
                                                                 mask);
                    }
                    StoreUnalignedFromFp32(yCurrent, yReg, yState, mask, active);
                }
                AscendC::MicroAPI::DataCopyUnAlignPost(yCurrent, yState, 0);
            }
        }
    }

    __aicore__ inline void ProcessChannelFirstTile(int64_t tileId, int64_t& cachedCBegin, int64_t& cachedCLen)
    {
        const int64_t b1Block = tileId % tilingData_->innerTileCount;
        const int64_t ncIndex = tileId / tilingData_->innerTileCount;
        const int64_t channel = ncIndex % tilingData_->c;
        const int64_t batch = ncIndex / tilingData_->c;
        const int64_t b1Begin = b1Block * tilingData_->tileElements;
        const int64_t count = tilingData_->inner - b1Begin < tilingData_->tileElements ? tilingData_->inner - b1Begin :
                                                                                         tilingData_->tileElements;
        const int64_t gmOffset = (batch * tilingData_->c + channel) * tilingData_->inner + b1Begin;
        EnsureParameters(channel, 1, cachedCBegin, cachedCLen);
        CopyInContiguous(gmOffset, count);
        ComputeChannelFirst(static_cast<uint16_t>(count));
        CopyOutContiguous(gmOffset, count);
    }

    __aicore__ inline void ProcessChannelLastTile(int64_t tileId, int64_t& cachedCBegin, int64_t& cachedCLen)
    {
        const int64_t bBlock = tileId % tilingData_->innerTileCount;
        const int64_t cBlock = tileId / tilingData_->innerTileCount;
        const int64_t bBegin = bBlock * tilingData_->tileRows;
        const int64_t cBegin = cBlock * tilingData_->tileElements;
        const int64_t rows = tilingData_->inner - bBegin < tilingData_->tileRows ? tilingData_->inner - bBegin :
                                                                                   tilingData_->tileRows;
        const int64_t cLen = tilingData_->c - cBegin < tilingData_->tileElements ? tilingData_->c - cBegin :
                                                                                   tilingData_->tileElements;
        const int64_t gmOffset = bBegin * tilingData_->c + cBegin;
        EnsureParameters(cBegin, cLen, cachedCBegin, cachedCLen);
        CopyInChannelLast(gmOffset, rows, cLen);
        ComputeChannelLast(static_cast<uint16_t>(rows), static_cast<uint16_t>(cLen));
        CopyOutChannelLast(gmOffset, rows, cLen);
    }

private:
    const BNInferenceTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_MEAN> meanGm_;
    GlobalTensor<T_VARIANCE> varianceGm_;
    GlobalTensor<T_MOMENTUM> momentumGm_;
    GlobalTensor<T_SCALE> scaleGm_;
    GlobalTensor<T_OFFSET> offsetGm_;
    GlobalTensor<T_X> yGm_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECIN, 1> meanQueue_;
    TQue<QuePosition::VECIN, 1> varianceQueue_;
    TQue<QuePosition::VECIN, 1> scaleQueue_;
    TQue<QuePosition::VECIN, 1> offsetQueue_;
    TBuf<TPosition::VECCALC> momentumBuffer_;
    TBuf<TPosition::VECCALC> meanCache_;
    TBuf<TPosition::VECCALC> rstdCache_;
    TBuf<TPosition::VECCALC> scaleCache_;
    TBuf<TPosition::VECCALC> offsetCache_;
    float factor_ = 0.0f;
};
} // namespace BNInferenceOps

#endif // BN_INFERENCE_GENERIC_H
