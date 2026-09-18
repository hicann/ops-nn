/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FUSED_MATMUL_SILU_KERNEL_H_
#define FUSED_MATMUL_SILU_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_matmul_silu_tiling_data.h"

namespace FusedMatmulSiluKernel {

using namespace AscendC;

constexpr MatmulConfig kMatmulConfig = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr uint8_t kCvSyncMode = 2;
constexpr uint64_t kCvPipelineDepth = 2;
constexpr uint16_t kCubeToVectorFlagBase = 0;
constexpr uint16_t kVectorToCubeFlagBase = 2;

struct MatmulTile {
    uint64_t mOffset = 0;
    uint64_t nOffset = 0;
    uint64_t mSize = 0;
    uint64_t nSize = 0;
};

template <typename T>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR,
                                const FusedMatmulSiluTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        weight_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
        bias_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if ASCEND_IS_AIC {
            matmul_.Init(&tiling_->matmulTiling, pipe_);
            matmul_.DisableBias();
        }
        if ASCEND_IS_AIV {
            uint32_t tileElems = tiling_->params.vectorTileElems;
            pipe_->InitBuffer(inputBuf_, tileElems * sizeof(T));
            pipe_->InitBuffer(biasBuf_, tileElems * sizeof(T));
            pipe_->InitBuffer(outputBuf_, tileElems * sizeof(T));
            pipe_->InitBuffer(valueBuf_, tileElems * sizeof(float));
            pipe_->InitBuffer(biasValueBuf_, tileElems * sizeof(float));
            pipe_->InitBuffer(tmpBuf_, tileElems * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessMatmul();
        }
        if ASCEND_IS_AIV {
            ProcessSilu();
        }
    }

private:
    using InputType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using WeightType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using OutputType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    __aicore__ inline uint64_t GetCoreIndex() const
    {
        uint64_t coreIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            int64_t taskRatio = GetTaskRation();
            if (taskRatio > 1) {
                coreIndex /= static_cast<uint64_t>(taskRatio);
            }
        }
        return coreIndex;
    }

    __aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) const
    {
        return divisor == 0 ? value : (value + divisor - 1) / divisor;
    }

    __aicore__ inline uint16_t GetCubeToVectorFlag(uint64_t step) const
    {
        return static_cast<uint16_t>(kCubeToVectorFlagBase + step % kCvPipelineDepth);
    }

    __aicore__ inline uint16_t GetVectorToCubeFlag(uint64_t step) const
    {
        return static_cast<uint16_t>(kVectorToCubeFlagBase + step % kCvPipelineDepth);
    }

    __aicore__ inline uint64_t GetMatmulTileCount(uint64_t& mTileCount, uint64_t& nTileCount) const
    {
        const TCubeTiling& matmulTiling = tiling_->matmulTiling;
        mTileCount = CeilDiv(tiling_->params.m, matmulTiling.singleCoreM);
        nTileCount = CeilDiv(tiling_->params.n, matmulTiling.singleCoreN);
        return mTileCount * nTileCount;
    }

    __aicore__ inline MatmulTile GetMatmulTile(uint64_t tileIndex, uint64_t nTileCount) const
    {
        if (nTileCount == 0) {
            return {};
        }
        const TCubeTiling& matmulTiling = tiling_->matmulTiling;
        uint64_t mTile = tileIndex / nTileCount;
        uint64_t nTile = tileIndex % nTileCount;
        MatmulTile tile;
        tile.mOffset = mTile * matmulTiling.singleCoreM;
        tile.nOffset = nTile * matmulTiling.singleCoreN;
        tile.mSize = (mTile + 1 == CeilDiv(tiling_->params.m, matmulTiling.singleCoreM)) ?
                         tiling_->params.m - tile.mOffset :
                         matmulTiling.singleCoreM;
        tile.nSize = (nTile + 1 == nTileCount) ? tiling_->params.n - tile.nOffset : matmulTiling.singleCoreN;
        return tile;
    }

    __aicore__ inline void ProcessMatmulTile(const MatmulTile& tile)
    {
        const TCubeTiling& matmulTiling = tiling_->matmulTiling;
        uint64_t xOffset = tile.mOffset * matmulTiling.Ka;
        uint64_t weightOffset = tile.nOffset * matmulTiling.Kb;
        uint64_t yOffset = tile.mOffset * tiling_->params.n + tile.nOffset;

        matmul_.SetOrgShape(tiling_->params.m, tiling_->params.n, tiling_->params.k);
        matmul_.SetSingleShape(tile.mSize, tile.nSize, tiling_->params.k);
        matmul_.SetTensorA(x_[xOffset], false);
        matmul_.SetTensorB(weight_[weightOffset], true);
        matmul_.template IterateAll<false>(y_[yOffset], 0, false);
        matmul_.End();
    }

    __aicore__ inline void ProcessMatmul()
    {
        uint64_t mTileCount = 0;
        uint64_t nTileCount = 0;
        uint64_t totalTiles = GetMatmulTileCount(mTileCount, nTileCount);
        uint64_t coreCount = tiling_->params.usedCoreNum;
        if (mTileCount == 0 || nTileCount == 0 || coreCount == 0) {
            return;
        }
        uint64_t coreIndex = GetCoreIndex();

        uint64_t localStep = 0;
        for (uint64_t tileIndex = coreIndex; tileIndex < totalTiles; tileIndex += coreCount, ++localStep) {
            if (localStep >= kCvPipelineDepth) {
                CrossCoreWaitFlag(GetVectorToCubeFlag(localStep - kCvPipelineDepth));
            }
            ProcessMatmulTile(GetMatmulTile(tileIndex, nTileCount));
            CrossCoreSetFlag<kCvSyncMode, PIPE_FIX>(GetCubeToVectorFlag(localStep));
        }
        uint64_t completedSteps = localStep > kCvPipelineDepth ? localStep - kCvPipelineDepth : 0;
        for (; completedSteps < localStep; ++completedSteps) {
            CrossCoreWaitFlag(GetVectorToCubeFlag(completedSteps));
        }
    }

    __aicore__ inline void ProcessSiluTile(const MatmulTile& tile, const LocalTensor<T>& input,
                                           const LocalTensor<T>& biasInput, const LocalTensor<T>& output,
                                           const LocalTensor<float>& value, const LocalTensor<float>& biasValue,
                                           const LocalTensor<float>& tmp)
    {
        uint64_t tileElems = tiling_->params.vectorTileElems;
        for (uint64_t row = 0; row < tile.mSize; ++row) {
            for (uint64_t offset = 0; offset < tile.nSize; offset += tileElems) {
                uint64_t remaining = tile.nSize - offset;
                uint32_t count = static_cast<uint32_t>(remaining < tileElems ? remaining : tileElems);
                uint32_t copyBytes = count * static_cast<uint32_t>(sizeof(T));
                uint64_t col = tile.nOffset + offset;
                uint64_t yOffset = (tile.mOffset + row) * tiling_->params.n + col;

                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                DataCopyPad(input, y_[yOffset], {1, copyBytes, 0, 0, 0}, {false, 0, 0, T(0)});
                DataCopyPad(biasInput, bias_[col], {1, copyBytes, 0, 0, 0}, {false, 0, 0, T(0)});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(value, input, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                Cast(biasValue, biasInput, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                Add(value, value, biasValue, count);
                PipeBarrier<PIPE_V>();
                Sigmoid(tmp, value, count);
                PipeBarrier<PIPE_V>();
                Mul(value, value, tmp, count);
                PipeBarrier<PIPE_V>();
                Cast(output, value, RoundMode::CAST_RINT, count);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(y_[yOffset], output, {1, copyBytes, 0, 0, 0});
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void ProcessSilu()
    {
        uint64_t mTileCount = 0;
        uint64_t nTileCount = 0;
        uint64_t totalTiles = GetMatmulTileCount(mTileCount, nTileCount);
        uint64_t coreCount = tiling_->params.usedCoreNum;
        if (tiling_->params.vectorTileElems == 0 || mTileCount == 0 || nTileCount == 0 || coreCount == 0) {
            return;
        }
        LocalTensor<T> input = inputBuf_.Get<T>();
        LocalTensor<T> biasInput = biasBuf_.Get<T>();
        LocalTensor<T> output = outputBuf_.Get<T>();
        LocalTensor<float> value = valueBuf_.Get<float>();
        LocalTensor<float> biasValue = biasValueBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        uint64_t localStep = 0;
        for (uint64_t tileIndex = GetCoreIndex(); tileIndex < totalTiles; tileIndex += coreCount, ++localStep) {
            CrossCoreWaitFlag(GetCubeToVectorFlag(localStep));
            ProcessSiluTile(GetMatmulTile(tileIndex, nTileCount), input, biasInput, output, value, biasValue, tmp);
            CrossCoreSetFlag<kCvSyncMode, PIPE_MTE3>(GetVectorToCubeFlag(localStep));
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    const FusedMatmulSiluTilingData* tiling_ = nullptr;
    TPipe* pipe_ = nullptr;
    matmul::MatmulImpl<InputType, WeightType, OutputType, BiasType, kMatmulConfig> matmul_;
    GlobalTensor<T> x_;
    GlobalTensor<T> weight_;
    GlobalTensor<T> bias_;
    GlobalTensor<T> y_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
    TBuf<TPosition::VECCALC> valueBuf_;
    TBuf<TPosition::VECCALC> biasValueBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
};

} // namespace FusedMatmulSiluKernel

#endif // FUSED_MATMUL_SILU_KERNEL_H_
