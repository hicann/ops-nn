/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_PATCH_MLP_H
#define FUSED_PATCH_MLP_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace FusedPatchMlp {

using namespace AscendC;

// Match the documented standard tanh GELU used by FusedPatchMlp.
constexpr float GELU_COEF_CUBIC = 0.044715f;
constexpr float GELU_ALPHA = -1.595769122f;
constexpr float SCALAR_ONE = 1.0f;
constexpr uint32_t GELU_MODE_ROW = 0U;
constexpr uint64_t DATA_BLOCK_BYTES = 32UL;
constexpr uint32_t GELU_BUFFER_NUM = 2U;
constexpr uint64_t SYNC_MODE_LOCAL_PAIR = 2UL;
constexpr uint64_t CUBE_TO_VECTOR_FLAG_BASE = 8UL;
constexpr uint64_t VECTOR_TO_CUBE_FLAG_BASE = 10UL;
// Match MatMulV3's base-kernel configuration. This class uses MatmulImpl directly, so SetSubBlockIdx and the unit
// flag are available even on CANN 9.1; the earlier restriction applied only to the registration-based wrapper.
constexpr MatmulConfig FUSED_PATCH_MLP_MDL_CFG = GetMDLConfig(false, false, 0, false, false, false, true);

template <typename T, bool USE_MDL, bool PIPELINE_GELU = false>
class KernelFusedPatchMlp {
public:
    using BiasT = typename std::conditional<std::is_same<T, bfloat16_t>::value, float, T>::type;
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, BiasT>;

    // MIX kernels in this repository initialize MatmulImpl explicitly. The registration-based Matmul wrapper is
    // intended for cube-only entry points and can run its registration path on AIV tasks as well.
    using DefaultMatmulType = matmul::MatmulImpl<AType, BType, CType, BiasType>;
    using MatmulType = typename std::conditional<
        USE_MDL, matmul::MatmulImpl<AType, BType, CType, BiasType, FUSED_PATCH_MLP_MDL_CFG>, DefaultMatmulType>::type;

    DefaultMatmulType mm0;
    MatmulType mmH;
    TPipe pipe;

    __aicore__ inline KernelFusedPatchMlp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weights, GM_ADDR biases, GM_ADDR y, GM_ADDR workspace,
                                const FusedPatchMlpTilingData* tiling);
    __aicore__ inline void Process();

private:
    template <typename MM>
    __aicore__ inline void RunTiledMatmul(MM& mm, GlobalTensor<T>& aGm, GlobalTensor<T>& cGm, uint64_t weightOffset,
                                          uint64_t biasOffset, uint32_t currentK, uint32_t tileM, uint32_t tileN,
                                          uint32_t usedCoreNum);
    template <typename MM>
    __aicore__ inline void RunTiledMatmulGelu(MM& mm, GlobalTensor<T>& aGm, GlobalTensor<T>& cGm, uint64_t weightOffset,
                                              uint64_t biasOffset, uint32_t currentK, uint32_t tileM, uint32_t tileN,
                                              uint32_t usedCoreNum);
    __aicore__ inline void GeluPass(const GlobalTensor<T>& src, const GlobalTensor<T>& dst);
    __aicore__ inline void GeluTile(const GlobalTensor<T>& src, const GlobalTensor<T>& dst, uint64_t offset,
                                    uint32_t count);
    __aicore__ inline void GeluTile2D(const GlobalTensor<T>& src, const GlobalTensor<T>& dst, uint32_t rowOffset,
                                      uint32_t rows, uint32_t colOffset, uint32_t cols);
    __aicore__ inline void GeluCompute(const LocalTensor<T>& input, const LocalTensor<T>& output, uint32_t count);

    // Match the standalone GELU kernel: two queue slots let MTE2/V/MTE3 overlap across adjacent tiles.
    TQue<QuePosition::VECIN, GELU_BUFFER_NUM> inQ_;
    TQue<QuePosition::VECOUT, GELU_BUFFER_NUM> outQ_;
    TBuf<TPosition::VECCALC> tmp1Buf_;
    TBuf<TPosition::VECCALC> tmp2Buf_;
    TBuf<TPosition::VECCALC> xfBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> wGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<BiasT> bGm_;
    GlobalTensor<T> ws0_;
    GlobalTensor<T> ws1_;

    uint32_t m_ = 0;
    uint32_t patch_ = 0;
    uint32_t hidden_ = 0;
    uint32_t geluTileSize_ = 0;
    uint32_t geluMode_ = GELU_MODE_ROW;
    uint32_t numLayers_ = 0;
    uint32_t mm0TileM_ = 0;
    uint32_t mm0TileN_ = 0;
    uint32_t mmHTileM_ = 0;
    uint32_t mmHTileN_ = 0;
    uint32_t mm0CoreNum_ = 1;
    uint32_t mmHCoreNum_ = 1;
};

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::Init(GM_ADDR x, GM_ADDR weights, GM_ADDR biases,
                                                                            GM_ADDR y, GM_ADDR workspace,
                                                                            const FusedPatchMlpTilingData* tiling)
{
    m_ = tiling->totalN;
    patch_ = tiling->inFeatures;
    hidden_ = tiling->hiddenSize;
    geluTileSize_ = tiling->geluTileSize;
    geluMode_ = tiling->geluMode;
    numLayers_ = tiling->numLayers;
    mm0TileM_ = tiling->mm0Tiling.singleCoreM;
    mm0TileN_ = tiling->mm0Tiling.singleCoreN;
    mmHTileM_ = tiling->mmHTiling.singleCoreM;
    mmHTileN_ = tiling->mmHTiling.singleCoreN;
    mm0CoreNum_ = tiling->mm0Tiling.usedCoreNum == 0U ? 1U : tiling->mm0Tiling.usedCoreNum;
    mmHCoreNum_ = tiling->mmHTiling.usedCoreNum == 0U ? 1U : tiling->mmHTiling.usedCoreNum;
    mm0.Init(&tiling->mm0Tiling, &pipe);
    if (numLayers_ > 1U) {
        if constexpr (USE_MDL) {
            mmH.SetSubBlockIdx(0);
        }
        mmH.Init(&tiling->mmHTiling, &pipe);
    }
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weights));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biases));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    // A single-layer MLP has no activation, so it does not need any vector UB buffers.
    if (numLayers_ <= 1) {
        return;
    }

    // GetUserWorkspace skips the framework-owned Matmul system workspace.
    __gm__ uint8_t* interBase = reinterpret_cast<__gm__ uint8_t*>(GetUserWorkspace(workspace));
    const uint64_t bufferElems = static_cast<uint64_t>(m_) * hidden_;
    ws0_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(interBase));
    ws1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(interBase + bufferElems * sizeof(T)));
    if (geluTileSize_ == 0) {
        geluTileSize_ = hidden_; // Backward compatibility for manually constructed tiling data.
    }

#ifdef __CCE_KT_TEST__
    if (true) {
#else
    // Matmul owns the AIC local-memory pipeline. GELU queues and temporary tensors are used only by AIV tasks.
    // Initializing them on AIC as well needlessly reserves UB and can interfere with Matmul's registered buffers.
    if ASCEND_IS_AIV {
#endif
        pipe.InitBuffer(inQ_, GELU_BUFFER_NUM, geluTileSize_ * sizeof(T));
        pipe.InitBuffer(outQ_, GELU_BUFFER_NUM, geluTileSize_ * sizeof(T));
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            pipe.InitBuffer(tmp1Buf_, geluTileSize_ * sizeof(float));
            pipe.InitBuffer(tmp2Buf_, geluTileSize_ * sizeof(float));
            pipe.InitBuffer(xfBuf_, geluTileSize_ * sizeof(float));
        } else {
            pipe.InitBuffer(tmp1Buf_, geluTileSize_ * sizeof(T));
            pipe.InitBuffer(tmp2Buf_, geluTileSize_ * sizeof(T));
        }
    }
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
template <typename MM>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::RunTiledMatmul(
    MM& mm, GlobalTensor<T>& aGm, GlobalTensor<T>& cGm, uint64_t weightOffset, uint64_t biasOffset, uint32_t currentK,
    uint32_t tileM, uint32_t tileN, uint32_t usedCoreNum)
{
    if (tileM == 0U || tileN == 0U || usedCoreNum == 0U) {
        return;
    }

    const uint32_t mTileCount = (m_ + tileM - 1U) / tileM;
    const uint32_t nTileCount = (hidden_ + tileN - 1U) / tileN;
    const uint64_t totalTiles = static_cast<uint64_t>(mTileCount) * nTileCount;
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= usedCoreNum) {
        return;
    }

    // OrgShape describes the full GM row strides and is invariant for every manually scheduled tile in this
    // Matmul. MatmulV3 likewise sets it once and only refreshes SingleShape for tail blocks. The large target
    // shape is exactly divisible by its 512x256 outer tile, so both shape updates can stay outside the hot loop.
    mm.SetOrgShape(m_, hidden_, currentK);
    const bool uniformTiles = (m_ % tileM == 0U) && (hidden_ % tileN == 0U);
    if (uniformTiles) {
        mm.SetSingleShape(tileM, tileN, currentK);
    }

    // N-major assignment lets all M blocks reuse the current B slice through L2. In a MIX kernel GetBlockNum()
    // includes vector tasks on some runtimes, so stride by Matmul tiling's AIC-only usedCoreNum instead.
    for (uint64_t task = blockIdx; task < totalTiles; task += usedCoreNum) {
        const uint32_t nIndex = static_cast<uint32_t>(task / mTileCount);
        const uint32_t mIndex = static_cast<uint32_t>(task - static_cast<uint64_t>(nIndex) * mTileCount);
        const uint32_t mOffset = mIndex * tileM;
        const uint32_t nOffset = nIndex * tileN;
        const uint32_t currentM = (mOffset + tileM <= m_) ? tileM : (m_ - mOffset);
        const uint32_t currentN = (nOffset + tileN <= hidden_) ? tileN : (hidden_ - nOffset);

        if (!uniformTiles) {
            mm.SetSingleShape(currentM, currentN, currentK);
        }
        mm.SetTensorA(aGm[static_cast<uint64_t>(mOffset) * currentK]);
        mm.SetTensorB(wGm_[weightOffset + nOffset]);
        mm.SetBias(bGm_[biasOffset + nOffset]);
        mm.template IterateAll<false>(cGm[static_cast<uint64_t>(mOffset) * hidden_ + nOffset]);
    }
    // End() is the Matmul API's completion point. An additional PIPE_ALL barrier and SetAtomicNone are redundant
    // here because this kernel never enables atomic output, and they add a per-layer cube-side drain.
    mm.End();
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
template <typename MM>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::RunTiledMatmulGelu(
    MM& mm, GlobalTensor<T>& aGm, GlobalTensor<T>& cGm, uint64_t weightOffset, uint64_t biasOffset, uint32_t currentK,
    uint32_t tileM, uint32_t tileN, uint32_t usedCoreNum)
{
    if (tileM == 0U || tileN == 0U || usedCoreNum == 0U) {
        return;
    }
    const uint32_t mTileCount = (m_ + tileM - 1U) / tileM;
    const uint32_t nTileCount = (hidden_ + tileN - 1U) / tileN;
    const uint64_t totalTiles = static_cast<uint64_t>(mTileCount) * nTileCount;

    if ASCEND_IS_AIC {
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum) {
            return;
        }
        mm.SetOrgShape(m_, hidden_, currentK);
        const bool uniformTiles = (m_ % tileM == 0U) && (hidden_ % tileN == 0U);
        if (uniformTiles) {
            mm.SetSingleShape(tileM, tileN, currentK);
        }

        uint32_t produced = 0U;
        for (uint64_t task = blockIdx; task < totalTiles; task += usedCoreNum, ++produced) {
            const uint32_t pingPong = produced & 1U;
            // Two flag sets may be in flight. SYNC_MODE2 broadcasts one C2V event to both paired AIV tasks and
            // merges their V2C acknowledgements into one AIC event, so each reused ID is waited exactly once.
            if (produced >= 2U) {
                CrossCoreWaitFlag(VECTOR_TO_CUBE_FLAG_BASE + pingPong);
            }

            const uint32_t nIndex = static_cast<uint32_t>(task / mTileCount);
            const uint32_t mIndex = static_cast<uint32_t>(task - static_cast<uint64_t>(nIndex) * mTileCount);
            const uint32_t mOffset = mIndex * tileM;
            const uint32_t nOffset = nIndex * tileN;
            const uint32_t currentM = (mOffset + tileM <= m_) ? tileM : (m_ - mOffset);
            const uint32_t currentN = (nOffset + tileN <= hidden_) ? tileN : (hidden_ - nOffset);
            if (!uniformTiles) {
                mm.SetSingleShape(currentM, currentN, currentK);
            }
            mm.SetTensorA(aGm[static_cast<uint64_t>(mOffset) * currentK]);
            mm.SetTensorB(wGm_[weightOffset + nOffset]);
            mm.SetBias(bGm_[biasOffset + nOffset]);
            mm.template IterateAll<false>(cGm[static_cast<uint64_t>(mOffset) * hidden_ + nOffset]);
            CrossCoreSetFlag<SYNC_MODE_LOCAL_PAIR, PIPE_FIX>(CUBE_TO_VECTOR_FLAG_BASE + pingPong);
        }
        mm.End();

        // The oldest tasks were acknowledged when their flag IDs were reused; only the last two remain.
        const uint32_t firstPending = produced > 2U ? produced - 2U : 0U;
        for (uint32_t index = firstPending; index < produced; ++index) {
            const uint32_t pingPong = index & 1U;
            CrossCoreWaitFlag(VECTOR_TO_CUBE_FLAG_BASE + pingPong);
        }
        return;
    }

    if ASCEND_IS_AIV {
        const uint32_t taskRatio = GetTaskRation();
        const uint32_t parentBlock = taskRatio == 0U ? 0U : GetBlockIdx() / taskRatio;
        if (parentBlock >= usedCoreNum) {
            return;
        }
        const uint32_t subBlock = GetSubBlockIdx();
        uint32_t produced = 0U;
        for (uint64_t task = parentBlock; task < totalTiles; task += usedCoreNum, ++produced) {
            const uint32_t pingPong = produced & 1U;
            CrossCoreWaitFlag(CUBE_TO_VECTOR_FLAG_BASE + pingPong);

            const uint32_t nIndex = static_cast<uint32_t>(task / mTileCount);
            const uint32_t mIndex = static_cast<uint32_t>(task - static_cast<uint64_t>(nIndex) * mTileCount);
            const uint32_t mOffset = mIndex * tileM;
            const uint32_t nOffset = nIndex * tileN;
            const uint32_t currentM = (mOffset + tileM <= m_) ? tileM : (m_ - mOffset);
            const uint32_t currentN = (nOffset + tileN <= hidden_) ? tileN : (hidden_ - nOffset);
            const uint32_t firstHalfRows = (currentM + 1U) / 2U;
            const uint32_t localRowOffset = subBlock == 0U ? 0U : firstHalfRows;
            const uint32_t localRows = subBlock == 0U ? firstHalfRows : (currentM - firstHalfRows);
            GeluTile2D(cGm, cGm, mOffset + localRowOffset, localRows, nOffset, currentN);
            // PIPE_MTE3 places the notification after every queued copy-out without draining unrelated pipelines.
            CrossCoreSetFlag<SYNC_MODE_LOCAL_PAIR, PIPE_MTE3>(VECTOR_TO_CUBE_FLAG_BASE + pingPong);
        }
    }
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::Process()
{
    if (m_ == 0 || numLayers_ == 0) {
        return;
    }

    GlobalTensor<T> aGm = xGm_;
    uint32_t currentK = patch_;
    uint64_t weightOffset = 0;
    uint64_t biasOffset = 0;

    for (uint32_t layer = 0; layer < numLayers_; ++layer) {
        const bool lastLayer = (layer == numLayers_ - 1);
        // Ping-pong Matmul outputs across ws0/ws1 so its input and output never alias. GELU can safely overwrite the
        // just-produced buffer in place: GeluTile completes MTE2 before vector compute and MTE3 writes the same tile,
        // while different AIV tasks own disjoint ranges. This avoids copying every activation into the other 40 MiB
        // buffer and reduces L2 cache-line churn for deep MLPs.
        GlobalTensor<T> cGm = lastLayer ? yGm_ : ((layer & 1U) == 0U ? ws0_ : ws1_);

#ifdef __CCE_KT_TEST__
        // The CPU kernel simulator has one unified core and cannot model AIC/AIV cross-core flags.
        if (layer == 0) {
            RunTiledMatmul(mm0, aGm, cGm, weightOffset, biasOffset, currentK, mm0TileM_, mm0TileN_, mm0CoreNum_);
        } else {
            RunTiledMatmul(mmH, aGm, cGm, weightOffset, biasOffset, currentK, mmHTileM_, mmHTileN_, mmHCoreNum_);
        }
        if (!lastLayer) {
            GeluPass(cGm, cGm);
            // The queue chain already orders MTE2 -> V -> MTE3. Only the final GM write must complete before the
            // following Matmul consumes this activation; draining every local pipeline is unnecessary.
            PipeBarrier<PIPE_MTE3>();
        }
#else
        const bool useTilePipeline = PIPELINE_GELU && layer > 0U && !lastLayer;
        if (useTilePipeline) {
            // Hidden-layer Matmul and GELU share the same 512x256 GM tile. Two paired AIV tasks consume the previous
            // tile while their AIC computes the next one; all local acknowledgements complete before this sole
            // layer-level barrier releases the following Matmul.
            RunTiledMatmulGelu(mmH, aGm, cGm, weightOffset, biasOffset, currentK, mmHTileM_, mmHTileN_, mmHCoreNum_);
            SyncAll<false>();
        } else {
            if ASCEND_IS_AIC {
                if (layer == 0) {
                    RunTiledMatmul(mm0, aGm, cGm, weightOffset, biasOffset, currentK, mm0TileM_, mm0TileN_,
                                   mm0CoreNum_);
                } else {
                    RunTiledMatmul(mmH, aGm, cGm, weightOffset, biasOffset, currentK, mmHTileM_, mmHTileN_,
                                   mmHCoreNum_);
                }
            }

            if (!lastLayer) {
                SyncAll<false>();
                if ASCEND_IS_AIV {
                    GeluPass(cGm, cGm);
                    PipeBarrier<PIPE_MTE3>();
                }
                SyncAll<false>();
            }
        }
#endif

        aGm = cGm;
        weightOffset += static_cast<uint64_t>(currentK) * hidden_;
        biasOffset += hidden_;
        currentK = hidden_;
    }
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::GeluCompute(const LocalTensor<T>& input,
                                                                                   const LocalTensor<T>& output,
                                                                                   uint32_t count)
{
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        LocalTensor<float> inputFp32 = xfBuf_.Get<float>();
        LocalTensor<float> temp1Fp32 = tmp1Buf_.Get<float>();
        LocalTensor<float> temp2Fp32 = tmp2Buf_.Get<float>();
        Cast(inputFp32, input, RoundMode::CAST_NONE, count);
        Mul(temp1Fp32, inputFp32, inputFp32, count);
        Mul(temp1Fp32, temp1Fp32, inputFp32, count);
        Muls(temp1Fp32, temp1Fp32, GELU_COEF_CUBIC, count);
        Add(temp1Fp32, temp1Fp32, inputFp32, count);
        Muls(temp1Fp32, temp1Fp32, GELU_ALPHA, count);
        Exp(temp2Fp32, temp1Fp32, count);
        Adds(temp2Fp32, temp2Fp32, SCALAR_ONE, count);
        // The converted input is dead after the denominator has been formed. AscendC vector division supports
        // dst/src0 aliasing, so keep the quotient in that tensor instead of reserving a fourth FP32 tile.
        Div(inputFp32, inputFp32, temp2Fp32, count);
        Cast(output, inputFp32, RoundMode::CAST_RINT, count);
    } else {
        LocalTensor<T> temp1 = tmp1Buf_.Get<T>();
        LocalTensor<T> temp2 = tmp2Buf_.Get<T>();
        Mul(temp1, input, input, count);
        Mul(temp1, temp1, input, count);
        Muls(temp1, temp1, static_cast<T>(GELU_COEF_CUBIC), count);
        Add(temp1, temp1, input, count);
        Muls(temp1, temp1, static_cast<T>(GELU_ALPHA), count);
        Exp(temp2, temp1, count);
        Adds(temp2, temp2, static_cast<T>(SCALAR_ONE), count);
        Div(output, input, temp2, count);
    }
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::GeluTile(const GlobalTensor<T>& src,
                                                                                const GlobalTensor<T>& dst,
                                                                                uint64_t offset, uint32_t count)
{
    LocalTensor<T> input = inQ_.template AllocTensor<T>();
    const uint64_t copyBytes = static_cast<uint64_t>(count) * sizeof(T);
    if (copyBytes % DATA_BLOCK_BYTES == 0) {
        DataCopy(input, src[offset], count);
    } else {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(input, src[offset], copyInParams, padParams);
    }
    inQ_.EnQue(input);
    input = inQ_.template DeQue<T>();

    LocalTensor<T> output = outQ_.template AllocTensor<T>();
    GeluCompute(input, output, count);

    outQ_.EnQue(output);
    output = outQ_.template DeQue<T>();
    if (copyBytes % DATA_BLOCK_BYTES == 0) {
        DataCopy(dst[offset], output, count);
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
        DataCopyPad(dst[offset], output, copyOutParams);
    }
    inQ_.FreeTensor(input);
    outQ_.FreeTensor(output);
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::GeluTile2D(const GlobalTensor<T>& src,
                                                                                  const GlobalTensor<T>& dst,
                                                                                  uint32_t rowOffset, uint32_t rows,
                                                                                  uint32_t colOffset, uint32_t cols)
{
    if (rows == 0U || cols == 0U) {
        return;
    }
    const uint32_t rowsPerUb = geluTileSize_ / cols;
    if (rowsPerUb == 0U) {
        // This path is not selected by the host for a tile wider than UB, but retain a safe fallback.
        for (uint32_t row = 0; row < rows; ++row) {
            GeluTile(src, dst, static_cast<uint64_t>(rowOffset + row) * hidden_ + colOffset, cols);
        }
        return;
    }

    const uint32_t rowBytes = cols * sizeof(T);
    const uint32_t gmGapBytes = (hidden_ - cols) * sizeof(T);
    for (uint32_t localRow = 0; localRow < rows; localRow += rowsPerUb) {
        const uint32_t currentRows = (localRow + rowsPerUb <= rows) ? rowsPerUb :
                                                                      static_cast<uint32_t>(rows - localRow);
        const uint32_t count = currentRows * cols;
        const uint64_t gmOffset = static_cast<uint64_t>(rowOffset + localRow) * hidden_ + colOffset;

        LocalTensor<T> input = inQ_.template AllocTensor<T>();
        DataCopyExtParams copyInParams{static_cast<uint16_t>(currentRows), rowBytes, gmGapBytes, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(input, src[gmOffset], copyInParams, padParams);
        inQ_.EnQue(input);
        input = inQ_.template DeQue<T>();

        LocalTensor<T> output = outQ_.template AllocTensor<T>();
        GeluCompute(input, output, count);
        outQ_.EnQue(output);
        output = outQ_.template DeQue<T>();

        DataCopyExtParams copyOutParams{static_cast<uint16_t>(currentRows), rowBytes, 0, gmGapBytes, 0};
        DataCopyPad(dst[gmOffset], output, copyOutParams);
        inQ_.FreeTensor(input);
        outQ_.FreeTensor(output);
    }
}

template <typename T, bool USE_MDL, bool PIPELINE_GELU>
__aicore__ inline void KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU>::GeluPass(const GlobalTensor<T>& src,
                                                                                const GlobalTensor<T>& dst)
{
    const uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
#ifndef __CCE_KT_TEST__
    // A MIX launch creates GetTaskRation() vector sub-blocks for each logical AIC block. GetBlockIdx() on AIV is
    // already the physical vector-task index, so partition against the physical AIV count to use both sub-blocks.
    blockNum *= GetTaskRation();
#endif
    if (blockNum == 0) {
        blockNum = 1;
    }
    if (blockIdx >= blockNum) {
        return;
    }

    if (geluMode_ == GELU_MODE_ROW) {
        const uint32_t rowsPerCore = (m_ + blockNum - 1U) / blockNum;
        const uint32_t startRow = blockIdx * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > m_) {
            endRow = m_;
        }
        for (uint32_t row = startRow; row < endRow; ++row) {
            GeluTile(src, dst, static_cast<uint64_t>(row) * hidden_, hidden_);
        }
        return;
    }

    // Assign whole contiguous tiles instead of rows. This keeps all large-shape cores balanced when M or hidden has
    // an awkward size, and drastically reduces per-row queue/DataCopy setup overhead.
    const uint64_t totalElems = static_cast<uint64_t>(m_) * hidden_;
    const uint64_t totalTiles = (totalElems + geluTileSize_ - 1UL) / geluTileSize_;
    const uint64_t smallCoreTiles = totalTiles / blockNum;
    const uint64_t extraTiles = totalTiles % blockNum;
    const uint64_t currentCoreTiles = smallCoreTiles + (blockIdx < extraTiles ? 1UL : 0UL);
    if (currentCoreTiles == 0) {
        return;
    }
    const uint64_t firstTile = static_cast<uint64_t>(blockIdx) * smallCoreTiles +
                               (blockIdx < extraTiles ? blockIdx : extraTiles);
    const uint64_t start = firstTile * geluTileSize_;
    uint64_t end = start + currentCoreTiles * geluTileSize_;
    if (end > totalElems) {
        end = totalElems;
    }
    for (uint64_t offset = start; offset < end; offset += geluTileSize_) {
        const uint64_t remaining = end - offset;
        const uint32_t count = static_cast<uint32_t>(remaining < geluTileSize_ ? remaining : geluTileSize_);
        GeluTile(src, dst, offset, count);
    }
}

} // namespace FusedPatchMlp

#endif // FUSED_PATCH_MLP_H
