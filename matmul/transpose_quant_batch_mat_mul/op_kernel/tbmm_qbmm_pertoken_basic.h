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
 * \file quant_batch_matmul_v3_pertoken_basic.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_PERTOKEN_BASIC_H
#define QUANT_BATCH_MATMUL_V3_PERTOKEN_BASIC_H

#include "tbmm_qbmm_base.h"
#include "tbmm_qbmm_basic_epilogue.h"
#include "pp_matmul_w8a8_cube.h"

namespace AscendC {

template <FusedOpType OpType>
struct EpilogueTypeTraits;

template <>
struct EpilogueTypeTraits<FusedOpType::GELU_TANH> {
    using Type = EpilogueDequantGeluTanh;
};

template <>
struct EpilogueTypeTraits<FusedOpType::GELU_ERF> {
    using Type = EpilogueDequantGeluErf;
};

template <TemplateBasicTypeForClass>
class BmmDequantPertokenBasic {
public:
    __aicore__ inline BmmDequantPertokenBasic() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
                                GM_ADDR workSpace, const QuantBatchMatmulV3TilingData* __restrict tilingData,
                                TPipe* tPipe)
    {
        blockIdx_ = GetBlockIdx();
        blockIdx_ /= GetTaskRation();
        subBlockIdx_ = GetSubBlockIdx();
        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        pipe_ = tPipe;
        InitTilingData(tilingData);
        usePp_ = !bTrans;
        if (!usePp_) {
            mm_.SetSubBlockIdx(0);
            mm_.Init(&(tilingData->matmulTiling), pipe_);
            SetMatmulOrgShape();
        }

        InitGlobalBuffers(x1, x2, scale, bias, pertokenScale, y, workSpace);
        if (!usePp_) {
            InitLocalBuffers();
        } else {
            if ASCEND_IS_AIV {
                if (packNd_) {
                    if (subBlockIdx_ == 0) {
                        PackNdBToNz();
                        NotifyEvent<PIPE_MTE3>(0x3);
                    }
                }
                InitLocalBuffers();
            }
            if ASCEND_IS_AIC {
                if (packNd_) {
                    WaitEvent(0x3);
                    weightGm_ = packedB_;
                    AscendC::SyncAll();
                }
                ppCube_.Init(batch_, ka_, n_, baseM_, baseN_, l0cBuffers_, weightNz_);
            }
        }
        offsetWorkspaceC_ = BUFFER_NUM * blockIdx_ * baseM_ * baseN_;

        if (!usePp_) {
            block_.Init(tilingData);
            update_.template Init<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(&tilingData->matmulTiling,
                                                                                      block_.params_);
            // TBMM A/Y batch is embedded in the M stride.
            update_.SetBatch(batch_);
        }
        loop_ = 0; // all_gather_quant_batch_mat_mul.h循环调Init和Process，管理CV同步的计数器每次都要清零
    }

    __aicore__ inline void UpdateBatchOffset(uint64_t batchIndex, QBmmBlockOffset& offset)
    {
        // ====== Step C: TBMM 布局 (A/Y 转置, token_scale 已在 aclnn 侧转置成 [B,M]) ======
        //   A[M, B, K] (perm_x1=[1,0,2]): batch 内嵌到 M 的 stride → b 维跨步 = K
        //   W[B, K, N] (perm_x2=[0,1,2]): batch 外轴, 和 QBMM 一致
        //   Y[M, B, N] (perm_y=[1,0,2]): batch 内嵌到 M 的 stride → b 维跨步 = N
        //   token_scale: aclnn 侧已转置 [M,B]→[B,M], 和 QBMM 原始 pertokenScale[B,M] 一致
        //   channel_scale[B, N]: 每个 batch 独立, b 跨步 = N
        offset.offsetA = offset.offsetA + batchIndex * ka_;
        offset.offsetB = offset.offsetB + batchIndex * ka_ * n_;
        offset.offsetC = offset.offsetC + batchIndex * n_;
        offset.offsetPertoken = offset.offsetPertoken + batchIndex * m_; // [B,M] 布局: b 外轴
        offset.offsetScale = offset.offsetScale + batchIndex * n_;
        if (biasThreeDim_) {
            offset.offsetBias = offset.offsetBias + batchIndex * n_;
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        // 带batch场景只需要在最开始初始化offsetWorkspaceC_和控制同步的loop
        bool pongSwitch = false;
        offsetWorkspaceC_ = BUFFER_NUM * blockIdx_ * baseM_ * baseN_;
        loop_ = 0;
        uint64_t pingOffsetC = offsetWorkspaceC_;
        if (usePp_) {
            if (!weightNz_ && l0cBuffers_ > 1U) {
                ProcessPpNd(pingOffsetC);
            } else {
                ProcessPp(pingOffsetC);
            }
            End();
            return;
        }
        for (uint64_t batchIndex = 0; batchIndex < batch_; batchIndex++) {
            bool reverse = true;
            // 首块计算，兼容无L2cache切分场景，减少scalar计算
            block_.InitFirstTileBlockIndex();
            OneTileCompute(0, 0, pingOffsetC, batchIndex, pongSwitch);

            for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
                reverse = !reverse;
                for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
                    uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
                    if (mTileIndex > 0 || nTileIndex > 0) { // 跳过首块
                        block_.UpdateBlockCnt(mTileIndex, nTileIndex);
                        block_.InitBlockIndex();
                        OneTileCompute(mTileIndex, nTileIndex, pingOffsetC, batchIndex, pongSwitch);
                    }
                }
            }
        }

        End();
    }

    __aicore__ inline UPDATE_TYPE& GetUpdateObj() { return update_; }

private:
    __aicore__ inline void NotifyFixpipeEpilogue(uint32_t curM, uint32_t curN, uint32_t resultSlot)
    {
        const uint16_t v2cSyncFlag = V2C_PING_FLAG | resultSlot;
        const uint16_t c2vSyncFlag = C2V_PING_FLAG | resultSlot;
        if ASCEND_IS_AIC {
            if (++loop_ > BUFFER_NUM) {
                WaitEvent(v2cSyncFlag);
            }
            ppCube_.Store(mmOutGm_[offsetWorkspaceC_], curM, curN, resultSlot);
            NotifyEvent<PIPE_FIX>(c2vSyncFlag);
        }
        if ASCEND_IS_AIV {
            WaitEvent(c2vSyncFlag);
            auto epilogueParams = InitEpilogueParams(curM, curN);
            epilogue_.Process(epilogueParams);
            NotifyEvent<PIPE_MTE2>(v2cSyncFlag);
        }
    }

    __aicore__ inline void CoreTaskRange(uint64_t total, uint64_t& myStart, uint64_t& myCount)
    {
        const uint64_t cores = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        const uint64_t q = total / cores;
        const uint64_t r = total % cores;
        if (blockIdx_ < r) {
            myCount = q + 1;
            myStart = blockIdx_ * myCount;
        } else {
            myCount = q;
            myStart = r * (q + 1) + (blockIdx_ - r) * q;
        }
    }

    __aicore__ inline void FillPpNext(uint64_t nextTask, uint32_t mCount, uint32_t nCount)
    {
        const uint64_t nextBatch = nextTask / (static_cast<uint64_t>(mCount) * nCount);
        const uint32_t nextMn = nextTask % (mCount * nCount);
        const uint32_t nextM = nextMn / nCount * baseM_;
        const uint32_t nextN = nextMn % nCount * baseN_;
        ppNextM_ = (m_ - nextM < baseM_) ? m_ - nextM : baseM_;
        ppNextN_ = (n_ - nextN < baseN_) ? n_ - nextN : baseN_;
        ppNextA_ = (static_cast<uint64_t>(nextM) * batch_ + nextBatch) * ka_;
        if (weightNz_) {
            const uint32_t kAlign = (ka_ + 15U) / 16U * 16U;
            const uint32_t nAlign = (n_ + 31U) / 32U * 32U;
            ppNextB_ = nextBatch * static_cast<uint64_t>(kAlign) * nAlign + static_cast<uint64_t>(nextN) * kAlign;
        } else {
            ppNextB_ = nextBatch * static_cast<uint64_t>(ka_) * n_ + nextN;
        }
    }

    __aicore__ inline void RunPpTile(uint64_t batchIndex, uint32_t mStart, uint32_t nStart, uint64_t pingOffsetC,
                                     uint32_t& slot, bool hasNext)
    {
        const uint32_t curM = (m_ - mStart < baseM_) ? m_ - mStart : baseM_;
        const uint32_t curN = (n_ - nStart < baseN_) ? n_ - nStart : baseN_;
        offset_.offsetA = (static_cast<uint64_t>(mStart) * batch_ + batchIndex) * ka_;
        if (weightNz_) {
            const uint32_t kAlign = (ka_ + 15U) / 16U * 16U;
            const uint32_t nAlign = (n_ + 31U) / 32U * 32U;
            offset_.offsetB = batchIndex * static_cast<uint64_t>(kAlign) * nAlign +
                              static_cast<uint64_t>(nStart) * kAlign;
        } else {
            offset_.offsetB = batchIndex * static_cast<uint64_t>(ka_) * n_ + nStart;
        }
        offset_.offsetC = (static_cast<uint64_t>(mStart) * batch_ + batchIndex) * n_ + nStart;
        offset_.offsetPertoken = batchIndex * m_ + mStart;
        offset_.offsetScale = batchIndex * n_ + nStart;
        offsetWorkspaceC_ = pingOffsetC + slot * baseM_ * baseN_;
        ppHasNext_ = hasNext;
        BasicMMCVFusionCompute(curM, curN, C2V_PING_FLAG | slot, V2C_PING_FLAG | slot);
        slot ^= 1;
    }

    __aicore__ inline void RunPpNdGroup(uint64_t batchIndex, uint32_t mStart, uint32_t nStart, uint32_t nTiles,
                                        uint64_t pingOffsetC, uint32_t& slot)
    {
        const uint32_t curM = (m_ - mStart < baseM_) ? m_ - mStart : baseM_;
        offset_.offsetA = (static_cast<uint64_t>(mStart) * batch_ + batchIndex) * ka_;
        offset_.offsetB = batchIndex * static_cast<uint64_t>(ka_) * n_ + nStart;
        offset_.offsetPertoken = batchIndex * m_ + mStart;
        ppHasNext_ = false;
        ppNTiles_ = nTiles;
        ppAvailN_ = n_ - nStart;
        ppSkipCompute_ = false;
        for (uint32_t i = 0; i < nTiles; ++i) {
            const uint32_t n0 = nStart + i * baseN_;
            const uint32_t curN = (n_ - n0 < baseN_) ? n_ - n0 : baseN_;
            offset_.offsetC = (static_cast<uint64_t>(mStart) * batch_ + batchIndex) * n_ + n0;
            offset_.offsetScale = batchIndex * n_ + n0;
            offsetWorkspaceC_ = pingOffsetC + slot * baseM_ * baseN_;
            if (i > 0) {
                ppSkipCompute_ = true;
            }
            BasicMMCVFusionCompute(curM, curN, C2V_PING_FLAG | slot, V2C_PING_FLAG | slot);
            slot ^= 1U;
        }
        ppSkipCompute_ = false;
        ppNTiles_ = 0;
    }

    __aicore__ inline void ProcessPpNd(uint64_t pingOffsetC)
    {
        const uint32_t mCount = (m_ + baseM_ - 1) / baseM_;
        const uint32_t nCount = (n_ + baseN_ - 1) / baseN_;
        const uint32_t nPerGroup = l0cBuffers_ == 0 ? 1U : l0cBuffers_;
        const uint32_t groupCount = (nCount + nPerGroup - 1U) / nPerGroup;
        const uint64_t total = static_cast<uint64_t>(batch_) * mCount * groupCount;
        uint32_t slot = 0;
        for (uint64_t task = blockIdx_; task < total; task += usedCoreNum_) {
            const uint64_t batchIndex = task / (static_cast<uint64_t>(mCount) * groupCount);
            const uint32_t mg = task % (mCount * groupCount);
            const uint32_t mStart = mg / groupCount * baseM_;
            const uint32_t group = mg % groupCount;
            const uint32_t nStart = group * nPerGroup * baseN_;
            const uint32_t nTiles = (nCount - group * nPerGroup < nPerGroup) ? (nCount - group * nPerGroup) : nPerGroup;
            RunPpNdGroup(batchIndex, mStart, nStart, nTiles, pingOffsetC, slot);
        }
    }

    __aicore__ inline void ProcessPp(uint64_t pingOffsetC)
    {
        const uint32_t mCount = (m_ + baseM_ - 1) / baseM_;
        const uint32_t nCount = (n_ + baseN_ - 1) / baseN_;
        const uint64_t total = static_cast<uint64_t>(batch_) * mCount * nCount;
        uint32_t slot = 0;
        if (!weightNz_) {
            uint64_t myStart = 0;
            uint64_t myCount = 0;
            CoreTaskRange(total, myStart, myCount);
            for (uint64_t t = 0; t < myCount; ++t) {
                const uint64_t task = myStart + t;
                const uint64_t batchIndex = task / (static_cast<uint64_t>(mCount) * nCount);
                const uint32_t mn = task % (mCount * nCount);
                const uint32_t mStart = mn / nCount * baseM_;
                const uint32_t nStart = mn % nCount * baseN_;
                const bool hasNext = t + 1 < myCount;
                if (hasNext) {
                    FillPpNext(task + 1, mCount, nCount);
                }
                RunPpTile(batchIndex, mStart, nStart, pingOffsetC, slot, hasNext);
            }
            return;
        }
        for (uint64_t task = blockIdx_; task < total; task += usedCoreNum_) {
            const uint64_t batchIndex = task / (static_cast<uint64_t>(mCount) * nCount);
            const uint32_t mn = task % (mCount * nCount);
            const uint32_t mStart = mn / nCount * baseM_;
            const uint32_t nStart = mn % nCount * baseN_;
            const bool hasNext = task + usedCoreNum_ < total;
            if (hasNext) {
                FillPpNext(task + usedCoreNum_, mCount, nCount);
            }
            RunPpTile(batchIndex, mStart, nStart, pingOffsetC, slot, hasNext);
        }
    }

    __aicore__ inline void InitTilingData(const QuantBatchMatmulV3TilingData* tilingData)
    {
        isPerTensor_ = tilingData->params.isPerTensor;
        m_ = tilingData->matmulTiling.M;
        n_ = tilingData->matmulTiling.N;
        ka_ = tilingData->matmulTiling.Ka;

        l0cBuffers_ = tilingData->matmulTiling.dbL0C;
        if (l0cBuffers_ == 0) {
            l0cBuffers_ = 1;
        }
        baseM_ = tilingData->matmulTiling.baseM;
        baseN_ = tilingData->matmulTiling.baseN;
        batchA_ = tilingData->params.batchA;
        batchB_ = tilingData->params.batchB;
        batch_ = tilingData->params.batchC;
        biasThreeDim_ = tilingData->params.biasThreeDim;

        hasBias_ = tilingData->matmulTiling.isBias;
        biasDtype_ = tilingData->params.biasDtype;
        if (biasDtype_ == DT_INT32 || biasDtype_ == DT_FLOAT) {
            biasDtypeSize_ = sizeof(int32_t);
        } else {
            biasDtypeSize_ = sizeof(half);
        }

        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        ubTmpBuffer_ = tilingData->params.needUbBuffer;
        packNd_ = false;
        weightNz_ = true;
        if constexpr (DequantBmm::GetFormat(x2Format) != CubeFormat::NZ) {
            packNd_ = tilingData->params.isDoubleScale != 0;
            weightNz_ = packNd_;
        }
    }

    __aicore__ inline void SetMatmulOrgShape()
    {
        // ====== Step C: TBMM 布局 orgShape ======
        //   A[M, B, K]: M 行步长 = B*K (batch 内嵌)   → orgKa = batch_ * ka_
        //   W[B, K, N]: 和 QBMM 一致                 → orgKb = ka_
        //   Y[M, B, N]: M 行步长 = B*N              → orgN  = batch_ * n_
        const uint64_t orgKa = static_cast<uint64_t>(batch_) * ka_;
        const uint64_t orgN = static_cast<uint64_t>(batch_) * n_;
        if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ &&
                      DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
            mm_.SetOrgShape(DequantBmm::Align(m_, BMM_BLOCK_NUM), DequantBmm::Align(n_, BMM_BLOCK_NUM),
                            DequantBmm::Align(ka_, K0_INT8), DequantBmm::Align(ka_, K0_INT8), orgN);
        } else if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ) {
            mm_.SetOrgShape(DequantBmm::Align(m_, K0_INT8), n_, DequantBmm::Align(ka_, BMM_BLOCK_NUM), ka_, orgN);
        } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
            if constexpr (bTrans) {
                mm_.SetOrgShape(m_, DequantBmm::Align(n_, BMM_BLOCK_NUM), orgKa, DequantBmm::Align(ka_, K0_INT8), orgN);
            } else {
                mm_.SetOrgShape(m_, DequantBmm::Align(n_, BMM_BLOCK_NUM), orgKa, DequantBmm::Align(ka_, BMM_BLOCK_NUM),
                                orgN);
            }
        } else {
            // ND x1, ND x2: TBMM A 的 M 维跨 batch 步长 orgKa=B*K; Y 的 M 维跨 batch 步长 orgN=B*N
            mm_.SetOrgShape(m_, n_, orgKa, ka_, orgN);
        }
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale,
                                             GM_ADDR y, GM_ADDR workSpace)
    {
        if (isPerTensor_) {
            scaleScalar_ = *((__gm__ scaleType*)scale);
        }
        xGm_.SetGlobalBuffer((__gm__ x1Type*)x1);
        weightGm_.SetGlobalBuffer((__gm__ x2Type*)x2);
        // ND B is strided [K, N] and the same batch is reread by extra N tiles.
        // Keep it L2-cacheable. NZ weights are streamed once per tile.
        if (m_ <= baseM_ && weightNz_) {
            weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (hasBias_ != 0U) {
            if (biasDtype_ == DT_BF16) {
                biasGmBf16_.SetGlobalBuffer((__gm__ bfloat16_t*)bias);
            } else if (biasDtype_ == DT_FLOAT16) {
                biasGmFp16_.SetGlobalBuffer((__gm__ half*)bias);
            } else if (biasDtype_ == DT_FLOAT) {
                biasGmFp32_.SetGlobalBuffer((__gm__ float*)bias);
            } else {
                biasGmInt32_.SetGlobalBuffer((__gm__ int32_t*)bias);
            }
        }
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        scaleGm_.SetGlobalBuffer((__gm__ scaleType*)scale);
        pertokenScaleGm_.SetGlobalBuffer((__gm__ float*)pertokenScale);
        mmOutGm_.SetGlobalBuffer((__gm__ int32_t*)workSpace, BUFFER_NUM * usedCoreNum_ * baseM_ * baseN_);
        if constexpr (DequantBmm::GetFormat(x2Format) != CubeFormat::NZ) {
            const uint64_t cBytes = static_cast<uint64_t>(BUFFER_NUM) * usedCoreNum_ * baseM_ * baseN_ *
                                    sizeof(int32_t);
            packedB_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workSpace + cBytes));
        }
    }

    __aicore__ inline void PackNdBToNz()
    {
        const uint32_t kAlign = (ka_ + 15U) / 16U * 16U;
        const uint32_t nAlign = (n_ + 31U) / 32U * 32U;
        const uint32_t n1 = nAlign / 32U;
        const uint32_t kPanel = 80;
        const uint32_t kTiles = (ka_ + kPanel - 1U) / kPanel;
        const uint64_t total = static_cast<uint64_t>(batch_) * kTiles;
        PpMatMulNS::OnChipBuffer<PpMatMulNS::ArchType::ASCEND_V220> buf;
        auto ub = buf.template GetBuffer<PpMatMulNS::BufferType::ASCEND_UB, int8_t>(0);
        auto in = ub[0];
        auto out = ub[kPanel * nAlign];
        for (uint64_t task = blockIdx_; task < total; task += usedCoreNum_) {
            const uint32_t batchIndex = task / kTiles;
            const uint32_t k0 = (task % kTiles) * kPanel;
            const uint32_t copyK = (ka_ - k0 < kPanel) ? ka_ - k0 : kPanel;
            AscendC::DataCopyPad(
                in, weightGm_[static_cast<uint64_t>(batchIndex) * ka_ * n_ + static_cast<uint64_t>(k0) * n_],
                {1, static_cast<uint32_t>(copyK * n_ * sizeof(int8_t)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            const uint16_t srcRowGap = static_cast<uint16_t>(n_ / 32U - 1U);
            for (uint32_t i = 0; i < n1; ++i) {
                AscendC::DataCopy(out[i * copyK * 32U], in[i * 32U], {static_cast<uint16_t>(copyK), 1, srcRowGap, 0});
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(
                packedB_[static_cast<uint64_t>(batchIndex) * kAlign * nAlign + static_cast<uint64_t>(k0) * 32U], out,
                {static_cast<uint16_t>(n1), static_cast<uint16_t>(copyK), 0, static_cast<uint16_t>(kAlign - copyK)});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
        pipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
        if (biasDtype_ != DT_INT32) {
            pipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
        }
        if (!isPerTensor_) {
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));
        }
        pipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
        pipe_->InitBuffer(vecQuePertokenScale_, BUFFER_NUM, DequantBmm::Align(ubCalcM_, 8U) * sizeof(float));
        pipe_->InitBuffer(broadcastFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    }

    __aicore__ inline void OneTileCompute(uint64_t mTileIndex, uint64_t nTileIndex, uint64_t pingOffsetC,
                                          uint64_t batchIndex, bool& pongSwitch)
    {
        for (uint64_t j = 0; j < block_.realRound_; j++) {
            // 更新此次基本块的大小和输入输出地址
            update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(
                block_.params_, offset_, mTileIndex, nTileIndex);

            UpdateBatchOffset(batchIndex, offset_);
            offsetWorkspaceC_ = pingOffsetC + pongSwitch * baseM_ * baseN_;
            BasicMMCVFusionCompute(block_.params_.singleCoreM, block_.params_.singleCoreN, C2V_PING_FLAG | pongSwitch,
                                   V2C_PING_FLAG | pongSwitch);
            pongSwitch = !pongSwitch;
            block_.UpdateBlockIndex();
        }
    }

    __aicore__ inline EpilogueParams<yType, scaleType> InitEpilogueParams(uint32_t CurAicM, uint32_t CurAicN)
    {
        return EpilogueParams<yType, scaleType>(
            mmOutGm_, yGm_, biasGmBf16_, biasGmFp16_, biasGmFp32_, scaleGm_,
            pertokenScaleGm_, // GlobalTensor

            outFp32Tmp_, vecQueTmp_, biasFp32Tmp_,
            broadcastFp32Tmp_, // TBuf

            vecQueSrc_, vecQueOut_, vecQueBias_, vecQueScale_,
            vecQuePertokenScale_, // TQue

            CurAicM, CurAicN, subBlockIdx_, ubCalcM_, ubCalcN_, offsetWorkspaceC_, biasDtype_, biasDtypeSize_,
            isPerTensor_, scaleScalar_, offset_, n_,
            batch_ // Step C: TBMM Y 行 stride = n_ * batch_, token_scale 跨步 batch_
        );
    }

    __aicore__ inline void BasicMMCVFusionCompute(uint32_t CurAicM, uint32_t CurAicN, uint16_t v2cSyncFlag,
                                                  uint16_t c2vSyncFlag)
    {
        if ASCEND_IS_AIC {
            const bool reuseGmSlot = ++loop_ > BUFFER_NUM;
            if (usePp_) {
                PrepareAndComputeMatmul(CurAicM, CurAicN);
                if (reuseGmSlot) {
                    WaitEvent(v2cSyncFlag);
                }
            } else if constexpr (bTrans) {
                PrepareAndComputeMatmul(CurAicM, CurAicN);
                if (reuseGmSlot) {
                    WaitEvent(v2cSyncFlag);
                }
            } else {
                if (reuseGmSlot) {
                    WaitEvent(v2cSyncFlag);
                }
                PrepareAndComputeMatmul(CurAicM, CurAicN);
            }
            if (usePp_) {
                ppCube_.Store(mmOutGm_[offsetWorkspaceC_], CurAicM, CurAicN, (loop_ - 1) % l0cBuffers_);
            } else {
                mm_.GetTensorC(mmOutGm_[offsetWorkspaceC_], 0, true);
            }
            NotifyEvent<PIPE_FIX>(c2vSyncFlag);
        }

        if ASCEND_IS_AIV {
            WaitEvent(c2vSyncFlag);
            auto epilogueParams = InitEpilogueParams(CurAicM, CurAicN);
            epilogue_.Process(epilogueParams);
            NotifyEvent<PIPE_MTE2>(v2cSyncFlag);
        }
    }

    __aicore__ inline void PrepareAndComputeMatmul(uint32_t baseM, uint32_t baseN)
    {
        if (usePp_) {
            if (ppSkipCompute_) {
                return;
            }
            const uint32_t resultSlot = (loop_ - 1) % l0cBuffers_;
            if (!weightNz_ && ppNTiles_ > 0) {
                ppCube_.ComputeAllN(xGm_[offset_.offsetA], weightGm_[offset_.offsetB], baseM, baseN, ppNTiles_,
                                    ppAvailN_);
                return;
            }
            ppCube_.Compute(xGm_[offset_.offsetA], weightGm_[offset_.offsetB], baseM, baseN, resultSlot);
            if (ppHasNext_) {
                ppCube_.Prefetch(xGm_[ppNextA_], weightGm_[ppNextB_], ppNextM_, ppNextN_);
            }
            return;
        }
        if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ && !bTrans) {
            mm_.SetOrgShape(m_, baseN, static_cast<uint64_t>(batch_) * ka_, DequantBmm::Align(ka_, BMM_BLOCK_NUM),
                            static_cast<uint64_t>(batch_) * n_);
        }
        mm_.SetSingleShape(baseM, baseN, ka_);
        mm_.SetTail(baseM, baseN, ka_);
        mm_.SetTensorA(xGm_[offset_.offsetA], aTrans);
        mm_.SetTensorB(weightGm_[offset_.offsetB], bTrans);

        if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
            mm_.SetBias(biasGmInt32_[offset_.offsetBias]);
        }
        mm_.Iterate();
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            // AIC跳过前两次Wait，也就是一次ping一次pong，这里补上
            if (loop_ > 0) { // 大于0表示需要补上开头跳过的ping
                WaitEvent(C2V_PING_FLAG);
            }

            if (loop_ > 1) { // 大于1表示需要补上开头跳过的pong
                WaitEvent(C2V_PONG_FLAG);
            }
            if (usePp_) {
                ppCube_.End();
            } else {
                mm_.End();
            }
        }
    }

private:
    GlobalTensor<x1Type> xGm_;
    GlobalTensor<x2Type> weightGm_;
    GlobalTensor<int32_t> biasGmInt32_;
    GlobalTensor<bfloat16_t> biasGmBf16_;
    GlobalTensor<half> biasGmFp16_;
    GlobalTensor<float> biasGmFp32_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<scaleType> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<int8_t> packedB_;

    TPipe* pipe_;
    // define the que
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    // used when bias type is bf16/fp16/fp32, dequant result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;

    scaleType scaleScalar_;

    // tilling data
    bool isPerTensor_;
    uint32_t usedCoreNum_;
    uint32_t m_;
    uint32_t n_;
    uint32_t ka_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t hasBias_;
    uint32_t biasDtype_ = 0;
    uint32_t biasDtypeSize_ = 0;
    uint32_t batch_ = 1;
    uint32_t batchA_ = 1;
    uint32_t batchB_ = 1;
    uint32_t biasThreeDim_;
    // vector
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;

    uint32_t blockIdx_;
    uint32_t subBlockIdx_ = 0;
    uint64_t offsetWorkspaceC_ = 0;
    uint64_t loop_ = 0;
    bool usePp_ = false;
    bool packNd_ = false;
    bool weightNz_ = true;
    uint32_t l0cBuffers_ = 2;
    bool ppHasNext_ = false;
    bool ppSkipCompute_ = false;
    uint32_t ppPairN_ = 0;
    uint32_t ppNTiles_ = 0;
    uint32_t ppAvailN_ = 0;
    uint32_t ppNextM_, ppNextN_;
    uint64_t ppNextA_, ppNextB_;
    PpMatMulNS::PpMatmulW8A8Cube<DequantBmm::GetFormat(x2Format) == CubeFormat::NZ> ppCube_;

    QuantBatchMatmulV3BaseBlock block_;
    UPDATE_TYPE update_; // 量化mm或mc2的更新计算大小和地址的接口
    QBmmBlockOffset offset_;
    EpilogueType epilogue_;

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, x1Type, aTrans>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, DequantBmm::GetFormat(x2Format), x2Type, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the TPos of ctype must be ub given by mm api when iterate<false>,
    // but actually we can move data to gm then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    static constexpr MatmulConfig MATMUL_CFG = bTrans ? TBMM_TRANSB_MDL_CFG : MM_DEFAULT_MDL_CFG;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MATMUL_CFG> mm_;
};

} // namespace AscendC

#endif // QUANT_BATCH_MATMUL_V3_PERTOKEN_BASIC_H
