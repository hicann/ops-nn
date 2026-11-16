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
 * \file quant_batch_matmul_v4_pergroup_fixpipe.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "tool.h"

#include "matmul_custom_impl.h"

using AscendC::BLOCK_CUBE;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using matmul::MatmulImpl;
using matmul::MatmulType;

#define LOCAL_TEMPLATE_CLASS_PARAMS \
    template <typename xType, typename wType, typename yType, bool bTrans, typename x1ScaleType, typename x2ScaleType>

#define LOCAL_TEMPLATE_FUNC_PARAMS xType, wType, yType, bTrans, x1ScaleType, x2ScaleType

namespace QuantBatchMatmulV4 {

template <typename yType>
struct AccParam {
    uint64_t mCount;

    __local_mem__ yType* addBaseAddr;
    __local_mem__ float* accBaseAddr;
    __local_mem__ float* addResBaseAddr;
};

template <typename yType>
struct ScaleParam {
    uint64_t mCount;

    __local_mem__ float* scaleBaseAddr;
    __local_mem__ yType* destUbAddr;
    __local_mem__ float* accBaseAddr;
};

LOCAL_TEMPLATE_CLASS_PARAMS
class QuantBatchMatmulV4PerGroupFixpipeKernel {
public:
    __aicore__ inline QuantBatchMatmulV4PerGroupFixpipeKernel(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y,
                                GM_ADDR workspace,
                                const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void InitUbBuffs();
    __aicore__ inline void ProcessSingleBlock(uint64_t kIterateIdx, uint64_t mOffset);
    __aicore__ inline void CopyResToGM(uint64_t mOffset, uint64_t nOffset);

    __aicore__ inline void InitInputs(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale,
                                      GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitL1Buffs();

    __aicore__ inline void GetABToL1(uint64_t mOffset, uint64_t nOffset, uint64_t kOffset, uint64_t kIterateIdx);
    __aicore__ inline void CopyX1Nd2Nz(uint64_t mOffset, uint64_t kOffset);
    __aicore__ inline void CopyX2Gm2L1(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyX2ScaleGm2L1(uint64_t nOffset, uint64_t kIterateIdx);

    __aicore__ inline void IterateMatmul(uint64_t kInnerIdx);

    __aicore__ inline void UpdateMmParams(uint64_t mOffset, uint64_t nOffset);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBaseBlockCompute(uint64_t mOffset, uint64_t nOffset, uint64_t totalKBlocks);
    __aicore__ inline void ProcessX1ScaleQuant(uint64_t mOffset, uint64_t nOffset, uint64_t kIterateIdx);

public:
    using inputWType = MatmulType<TPosition::TSCM, CubeFormat::NZ, DTYPE_X2, bTrans, LayoutMode::NONE, false,
                                  TPosition::GM>;
    MatmulCustomImpl<xType, wType, yType, yType, x1ScaleType, x2ScaleType, false, bTrans, inputWType> mmObj_;
    TPipe* pipe_;
    const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tiling_;

protected:
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;

    int32_t curBlockIdx_;

    uint64_t curL1BufIdx_ = 0;
    // fixPipe到UB的buff index，开double buffer
    uint64_t curFixOutUbBufIdx_ = 0;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<uint8_t> wGlobalU8_;
    GlobalTensor<x1ScaleType> x1ScaleGlobal_;
    GlobalTensor<x2ScaleType> x2ScaleGlobal_;
    GlobalTensor<yType> yGlobal_;

    // A、B矩阵L1默认开Double Buffer
    LocalTensor<xType> aL1PingLocalBuf_;
    LocalTensor<xType> aL1PongLocalBuf_;
    LocalTensor<wType> bL1PingLocalBuf_;
    LocalTensor<wType> bL1PongLocalBuf_;
    LocalTensor<x2ScaleType> bScaleL1PingBuf_;
    LocalTensor<x2ScaleType> bScaleL1PongBuf_;

    uint64_t bScaleL1DataSize_;

    // MTE1_MTE2同步：A、B矩阵用同一组
    TEventID eventIdL1PingMte1ToMte2_;
    TEventID eventIdL1PongMte1ToMte2_;

    // X2 Scale FIX和MTE2同步
    TEventID eventIdL1PingFixToMte2_;
    TEventID eventIdL1PongFixToMte2_;

    // FixPipe等Vector的核间同步总次数
    uint64_t pipeVToFixSyncCount_;

    // 放在bank0的fixpipe结果
    LocalTensor<yType> addPingLocalTensor_;
    // 放在bank1的累加和结果
    LocalTensor<float> accPongLocalTensor_;
    // 放在bank2的累加和结果
    LocalTensor<float> accPingLocalTensor_;
    // 放在bank3的fixpipe结果
    LocalTensor<yType> addPongLocalTensor_;

    LocalTensor<float> x1ScaleUbTensor_;
    LocalTensor<yType> resUbTensor_;

    uint64_t curMSize_;
    uint64_t curNSize_;
    uint64_t curMAlignSize_;
    uint64_t curNAlignSize_;
    uint64_t curKL1Size_;

    uint64_t kBlockSize_;
    uint64_t aL1BaseOffset_;
    uint64_t bL1BaseOffset_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace,
    const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tilingData, TPipe* tPipe)
{
    tiling_ = tilingData;
    pipe_ = tPipe;

    curBlockIdx_ = GetBlockIdx();

    kBlockSize_ = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;

    InitInputs(x1, x2, x1_scale, x2_scale, y_scale, y, workspace);
    InitL1Buffs();
    InitUbBuffs();

    mmObj_.Init(&tiling_->matmulTiling, tPipe);

    // 申请AL1和BL1 MTE1_MTE2同步事件ID
    if ASCEND_IS_AIC {
        eventIdL1PingMte1ToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        SetFlag<HardEvent::MTE1_MTE2>(eventIdL1PingMte1ToMte2_);
        eventIdL1PongMte1ToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        SetFlag<HardEvent::MTE1_MTE2>(eventIdL1PongMte1ToMte2_);

        eventIdL1PingFixToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE2>();
        SetFlag<HardEvent::FIX_MTE2>(eventIdL1PingFixToMte2_);
        eventIdL1PongFixToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE2>();
        SetFlag<HardEvent::FIX_MTE2>(eventIdL1PongFixToMte2_);
    }

    uint64_t baseKCount = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, tiling_->matmulTiling.baseK);
    pipeVToFixSyncCount_ = baseKCount > 2 ? baseKCount - 2 : 0; // 2为double buff，最后两次不用设置同步
}

/*
 *  功能：申请各输入的 global buffer
 */
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitInputs(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace)
{
    // 申请 Global Buffer
    if ASCEND_IS_AIC {
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
        wGlobalU8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x2));
        x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x2ScaleType*>(x2_scale));
    }

    if ASCEND_IS_AIV {
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
        x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x1ScaleType*>(x1_scale));
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitL1Buffs()
{
    if ASCEND_IS_AIV {
        return;
    }

    // 申请A、B矩阵、B Scale的L1 Buffer 空间
    TBuf<QuePosition::TSCM> l1Tbuf_;
    pipe_->InitBuffer(l1Tbuf_, L1_BUFFER_SIZE);

    // B矩阵L1 Buff大小，BL1是int4类型
    uint64_t bL1DataSizes = tiling_->matmulTiling.baseN * kBlockSize_;
    uint64_t bL1DataBytes = bL1DataSizes >> 1;

    // A矩阵L1 Buff大小
    uint64_t aL1DataSizes = tiling_->matmulTiling.baseM * kBlockSize_;
    uint64_t aL1DataBytes = aL1DataSizes * sizeof(xType);

    // B Scale矩阵L1 Buff大小
    bScaleL1DataSize_ = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepKb;

    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    // L1 ping buff分配
    bL1PingLocalBuf_ = l1Tbuf_.template GetWithOffset<wType>(bL1DataSizes, l1BufOffsetPart1);
    l1BufOffsetPart1 += bL1DataBytes;

    aL1PingLocalBuf_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSizes, l1BufOffsetPart1);
    l1BufOffsetPart1 += aL1DataBytes;

    bScaleL1PingBuf_ = l1Tbuf_.template GetWithOffset<x2ScaleType>(bScaleL1DataSize_, l1BufOffsetPart1);

    // L1 pong buff分配
    bL1PongLocalBuf_ = l1Tbuf_.template GetWithOffset<wType>(bL1DataSizes, l1BufOffsetPart2);
    l1BufOffsetPart2 += bL1DataBytes;

    aL1PongLocalBuf_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSizes, l1BufOffsetPart2);
    l1BufOffsetPart2 += aL1DataBytes;

    bScaleL1PongBuf_ = l1Tbuf_.template GetWithOffset<x2ScaleType>(bScaleL1DataSize_, l1BufOffsetPart2);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    uint64_t usedCoreNum = tiling_->cubeBlockDimM * tiling_->cubeBlockDimN;
    if (curBlockIdx_ >= usedCoreNum) {
        return;
    }

    // 计算总的块数量
    uint64_t totalMBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.M, tiling_->matmulTiling.baseM);
    uint64_t totalNBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.N, tiling_->matmulTiling.baseN);
    uint64_t totalBlocks = totalMBlocks * totalNBlocks;

    uint64_t totalKBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, kBlockSize_);
    // 当前核负责的块索引
    for (uint64_t blockIdx = curBlockIdx_; blockIdx < totalBlocks; blockIdx += usedCoreNum) {
        // 计算对应的 mOffset 和 nOffset
        uint64_t mOffset = (blockIdx / totalNBlocks) * tiling_->matmulTiling.baseM;
        uint64_t nOffset = (blockIdx % totalNBlocks) * tiling_->matmulTiling.baseN;

        UpdateMmParams(mOffset, nOffset);
        // 处理该块
        ProcessBaseBlockCompute(mOffset, nOffset, totalKBlocks);
    }

    if ASCEND_IS_AIC {
        mmObj_.End();

        WaitFlag<HardEvent::MTE1_MTE2>(eventIdL1PingMte1ToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdL1PingMte1ToMte2_);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdL1PongMte1ToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdL1PongMte1ToMte2_);

        WaitFlag<HardEvent::FIX_MTE2>(eventIdL1PingFixToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE2>(eventIdL1PingFixToMte2_);
        WaitFlag<HardEvent::FIX_MTE2>(eventIdL1PongFixToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE2>(eventIdL1PongFixToMte2_);
    }
}

__aicore__ inline void NotifyVector() { AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIV_AIC_FLAG); }

__aicore__ inline void WaitForCube() { AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(SYNC_AIV_AIC_FLAG); }

__aicore__ inline void WaitForVector() { AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_FLAG); }

__aicore__ inline void NotifyCube() { AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_V>(SYNC_AIC_AIV_FLAG); }

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateMmParams(
    uint64_t mOffset, uint64_t nOffset)
{
    if ASCEND_IS_AIC {
        curMSize_ = Min<uint64_t>(tiling_->matmulTiling.M - mOffset, tiling_->matmulTiling.baseM);
        curNSize_ = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
        curMAlignSize_ = CeilAlign<uint64_t>(curMSize_, BLOCK_CUBE);
        curNAlignSize_ = CeilAlign<uint64_t>(curNSize_, BLOCK_CUBE);
        // per-group baseK=group_size，tiling保证K方向不会带尾块，这里直接设置成baseK就可以
        mmObj_.SetTail(curMSize_, curNSize_, tiling_->matmulTiling.baseK);

        aL1BaseOffset_ = tiling_->matmulTiling.baseK * curMAlignSize_;
        bL1BaseOffset_ = tiling_->matmulTiling.baseK * 64; // int4 n0 size为64
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessBaseBlockCompute(
    uint64_t mOffset, uint64_t nOffset, uint64_t totalKBlocks)
{
    uint64_t kIterateIdx = 0;
    for (uint64_t kOffset = 0; kOffset < tiling_->matmulTiling.Ka; kOffset += kBlockSize_) {
        // A、B矩阵K方向搬入 L1 Size 计算
        curKL1Size_ = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
        if ASCEND_IS_AIC {
            GetABToL1(mOffset, nOffset, kOffset, kIterateIdx);

            mmObj_.SetOrgShape(curMAlignSize_, curNAlignSize_, curKL1Size_, curKL1Size_, tiling_->matmulTiling.baseN);

            // 等待GM搬入L1完成后启动Matmul计算
            event_t eventIDMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
            SetFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
            WaitFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
        }

        // k内部循环，因为需要在Vector累加，所以需要保证每次Matmul的K size在同一个group内
        uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKL1Size_, tiling_->matmulTiling.baseK);
        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            if ASCEND_IS_AIC {
                // FixPipe交替写入UB的两块buffer，前两次不用等Vector同步
                if (likely(kIterateIdx > 1)) {
                    WaitForVector();
                }

                IterateMatmul(kInnerIdx);
                NotifyVector();
            }

            if ASCEND_IS_AIV {
                WaitForCube();
                ProcessSingleBlock(kIterateIdx, mOffset);
                // FixPipe交替写入UB的两块buffer，最后两次baseK不需要设置核间同步标记防止不匹配
                if (likely(kIterateIdx < pipeVToFixSyncCount_)) {
                    NotifyCube();
                }
            }
            kIterateIdx++;
        }

        if ASCEND_IS_AIC {
            curL1BufIdx_ == IDX_0 ? SetFlag<HardEvent::MTE1_MTE2>(eventIdL1PingMte1ToMte2_) :
                                    SetFlag<HardEvent::MTE1_MTE2>(eventIdL1PongMte1ToMte2_);

            curL1BufIdx_ == IDX_0 ? SetFlag<HardEvent::FIX_MTE2>(eventIdL1PingFixToMte2_) :
                                    SetFlag<HardEvent::FIX_MTE2>(eventIdL1PongFixToMte2_);

            // 一次K方向计算结束，A和B矩阵切换对应的L1 Buff
            curL1BufIdx_ = curL1BufIdx_ ^ 1;
        }
    }

    if ASCEND_IS_AIV {
        // 基本块累加结束后，做x1_scale的量化
        ProcessX1ScaleQuant(mOffset, nOffset, kIterateIdx - 1);

        event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        // 将结果搬运到GM
        CopyResToGM(mOffset, nOffset);
        event_t eventIDMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
    }

    // 每轮K方向计算完，清空下FixOutUbBufIdx，下一轮计算FixPipe从ping buff开始写入，和VF保持一致
    curFixOutUbBufIdx_ = 0;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetABToL1(
    uint64_t mOffset, uint64_t nOffset, uint64_t kOffset, uint64_t kIterateIdx)
{
    // 等待L1 buffer可用
    curL1BufIdx_ == IDX_0 ? WaitFlag<HardEvent::MTE1_MTE2>(eventIdL1PingMte1ToMte2_) :
                            WaitFlag<HardEvent::MTE1_MTE2>(eventIdL1PongMte1ToMte2_);

    CopyX1Nd2Nz(mOffset, kOffset);

    // A、B共用1组同步，拷贝A前等待L1 buffer可用
    CopyX2Gm2L1(nOffset, kOffset);

    // X2 Scale提前搬入到L1给FixPipe使用
    CopyX2ScaleGm2L1(nOffset, kIterateIdx);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX1Nd2Nz(
    uint64_t mOffset, uint64_t kOffset)
{
    // A 矩阵在 GM 上搬运的偏移计算：mOffset * K + kOffset
    uint64_t aGmOffset = mOffset * tiling_->matmulTiling.Ka + kOffset;

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curMSize_;
    nd2nzParams.dValue = curKL1Size_;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;

    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    DataCopy((curL1BufIdx_ == IDX_0) ? aL1PingLocalBuf_ : aL1PongLocalBuf_, xGlobal_[aGmOffset], nd2nzParams, true);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX2Gm2L1(
    uint64_t nOffset, uint64_t kOffset)
{
    // 计算X2 GM偏移大小：int4需要除以2，kOffset * n0_size + nL1Offset * K
    uint64_t x2GmOffset = (kOffset * 64 + nOffset * tiling_->matmulTiling.Kb) / 2;

    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = CeilDiv<uint64_t>(curNSize_, ONE_BLK_ELEM_B4);
    dataCopyParams.blockLen = curKL1Size_;
    dataCopyParams.srcStride = tiling_->matmulTiling.Kb - curKL1Size_;
    dataCopyParams.dstStride = 0;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    AscendC::DataCopy((curL1BufIdx_ == IDX_0) ? bL1PingLocalBuf_.template ReinterpretCast<uint8_t>() :
                                                bL1PongLocalBuf_.template ReinterpretCast<uint8_t>(),
                      wGlobalU8_[x2GmOffset], dataCopyParams, true);
}

/**
 * @brief 将 X2 Scale 数据从 GM 拷贝到 L1，每次拷贝 nL1Size * stepK 的大小供 FixPipe 使用。
 */
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX2ScaleGm2L1(
    uint64_t nOffset, uint64_t kIterateIdx)
{
    uint64_t bScaleGmOffset = kIterateIdx * tiling_->matmulTiling.N + nOffset;
    constexpr uint64_t FIXPIPE_DEQ_DATA_SIZE = 16 * sizeof(x2ScaleType); // FIXPIPE_DEQ_TENSOR_SIZE 16元素对齐
    uint64_t deqDataSize = CeilAlign(curNSize_ * sizeof(uint64_t), FIXPIPE_DEQ_DATA_SIZE);
    uint16_t blockCount = static_cast<uint16_t>(CeilDiv<uint64_t>(curKL1Size_, tiling_->matmulTiling.baseK));

    // srcStride:(N - nOffSet - nL1Size) + nOffset = (N - nL1Size)
    uint16_t srcStride = static_cast<uint16_t>(((tiling_->matmulTiling.N - curNSize_) * sizeof(uint64_t)) /
                                               32); // 单位：32B
    AscendC::DataCopyParams intriParams{blockCount, static_cast<uint16_t>(deqDataSize / ONE_BLK_SIZE), srcStride, 0};

    // 等待L1 buffer可用
    (curL1BufIdx_ == IDX_0) ? WaitFlag<HardEvent::FIX_MTE2>(eventIdL1PingFixToMte2_) :
                              WaitFlag<HardEvent::FIX_MTE2>(eventIdL1PongFixToMte2_);
    DataCopy((curL1BufIdx_ == IDX_0) ? bScaleL1PingBuf_ : bScaleL1PongBuf_, x2ScaleGlobal_[bScaleGmOffset], intriParams,
             true);

    // FixPipe等bScale搬入到L1
    event_t eventIDMte2ToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_FIX));
    SetFlag<HardEvent::MTE2_FIX>(eventIDMte2ToFix);
    WaitFlag<HardEvent::MTE2_FIX>(eventIDMte2ToFix);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IterateMatmul(
    uint64_t kInnerIdx)
{
    if ASCEND_IS_AIC {
        uint64_t aL1Offset = kInnerIdx * aL1BaseOffset_;
        mmObj_.SetTensorA((curL1BufIdx_ == IDX_0) ? aL1PingLocalBuf_[aL1Offset] : aL1PongLocalBuf_[aL1Offset]);

        // kOffSet * n0_size
        uint64_t bL1Offset = kInnerIdx * bL1BaseOffset_;
        mmObj_.SetTensorB((curL1BufIdx_ == IDX_0) ? bL1PingLocalBuf_[bL1Offset] : bL1PongLocalBuf_[bL1Offset]);

        uint64_t bScaleL1Offset = kInnerIdx * curNSize_;
        mmObj_.SetQuantVector((curL1BufIdx_ == IDX_0) ? bScaleL1PingBuf_[bScaleL1Offset] :
                                                        bScaleL1PongBuf_[bScaleL1Offset]);

        mmObj_.Iterate(false);

        // Fixpipe结果需要交替写入UB
        curFixOutUbBufIdx_ == 0 ? mmObj_.GetTensorC(addPingLocalTensor_) : mmObj_.GetTensorC(addPongLocalTensor_);
        curFixOutUbBufIdx_ ^= 1;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitUbBuffs()
{
    // 第一个4指的是4个buffer，第二个4指的是buffer中的数据类型是float，每个float是4个字节
    uint32_t bankUbSize = 4 * tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 4;
    TBuf<QuePosition::VECIN> bankBuf;
    pipe_->InitBuffer(bankBuf, bankUbSize);
    addPingLocalTensor_ = bankBuf.template GetWithOffset<yType>(bankUbSize / 2, 0);
    accPongLocalTensor_ = bankBuf.template GetWithOffset<float>((bankUbSize - 256) / 4, 256);
    accPingLocalTensor_ = bankBuf.template GetWithOffset<float>((bankUbSize - 512) / 4, 512);
    addPongLocalTensor_ = bankBuf.template GetWithOffset<yType>((bankUbSize - 768) / 2, 768);

    TBuf<QuePosition::VECOUT> outBuf;
    pipe_->InitBuffer(outBuf, tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 2);
    resUbTensor_ = outBuf.Get<yType>();

    TBuf<QuePosition::VECIN> scaleBuf;
    pipe_->InitBuffer(scaleBuf, tiling_->matmulTiling.baseM * 4);
    x1ScaleUbTensor_ = scaleBuf.Get<float>();
}

// cast单个block并且搬运到目标地址
template <typename yType>
__aicore__ inline void CastSingleBlockVf(const AccParam<yType>& param)
{
    uint32_t nBfStride = 64;
    uint32_t mFloatStride = 256;
    uint32_t mBfStride = 512;
    uint32_t dataCopyStride = 448;
    MicroAPI::RegTensor<yType> addBf16RegTensor;
    MicroAPI::RegTensor<float> addFloatRegTensor;

    MicroAPI::MaskReg castAddCopyMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait castTraitNorm = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};
    __local_mem__ float* destUbAddr = param.addResBaseAddr;
    __local_mem__ yType* addUbAddr = param.addBaseAddr;

    for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
        // 将add数据拷贝到RegTensor中，每次拷贝64个数字，n循环的间隔是64B，m循环的间隔是1024B
        MicroAPI::DataCopy<yType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
            addBf16RegTensor, addUbAddr, nBfStride);
        // 将bf16转成float
        MicroAPI::Cast<float, yType, castTraitNorm>(addFloatRegTensor, addBf16RegTensor, castAddCopyMask);

        // 将cast结果拷出到UB
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(destUbAddr, addFloatRegTensor, mFloatStride,
                                                                           castAddCopyMask);

        // 将add的数据从UB拷入RegTensor
        MicroAPI::DataCopy<yType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
            addBf16RegTensor, addUbAddr, dataCopyStride);
        // 将bf16转成float
        MicroAPI::Cast<float, yType, castTraitNorm>(addFloatRegTensor, addBf16RegTensor, castAddCopyMask);

        // 将cast结果拷出到UB
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(destUbAddr, addFloatRegTensor, mFloatStride,
                                                                           castAddCopyMask);
    }
}

template <typename yType>
__aicore__ inline void AccSingleBlockVf(const AccParam<yType>& param)
{
    uint32_t nBfStride = 64;
    uint32_t mFloatStride = 256;
    uint32_t mBfStride = 512;
    uint32_t dataCopyStride = 448;
    MicroAPI::RegTensor<yType> addBf16RegTensor;
    MicroAPI::RegTensor<float> addFloatRegTensor;
    MicroAPI::RegTensor<float> accFloatRegTensor;
    MicroAPI::RegTensor<float> addResRegTensor;

    MicroAPI::MaskReg castAddCopyMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait castTraitNorm = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};
    __local_mem__ float* destUbAddr = param.addResBaseAddr;
    __local_mem__ float* accUbAddr = param.accBaseAddr;
    __local_mem__ yType* addUbAddr = param.addBaseAddr;

    for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
        // 将add数据拷贝到RegTensor中，每次拷贝64个数字，n循环的间隔是64B，m循环的间隔是1024B
        MicroAPI::DataCopy<yType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
            addBf16RegTensor, addUbAddr, nBfStride);

        // 将bf16转成float
        MicroAPI::Cast<float, yType, castTraitNorm>(addFloatRegTensor, addBf16RegTensor, castAddCopyMask);

        // 将ACC拷贝到RegTensor中，每次拷贝128个数字，每次之间的拷贝距离是1024B
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(accFloatRegTensor, accUbAddr, mFloatStride);

        // 累加求和
        MicroAPI::Add(addResRegTensor, accFloatRegTensor, addFloatRegTensor, castAddCopyMask);

        // 将结果拷贝到UB
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(destUbAddr, addResRegTensor, mFloatStride,
                                                                           castAddCopyMask);

        // 将add数据拷贝到RegTensor中，每次拷贝64个数字，n循环的间隔是64B，m循环的间隔是1024B
        MicroAPI::DataCopy<yType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
            addBf16RegTensor, addUbAddr, dataCopyStride);

        // 将bf16转成float
        MicroAPI::Cast<float, yType, castTraitNorm>(addFloatRegTensor, addBf16RegTensor, castAddCopyMask);

        // 将ACC拷贝到RegTensor中，每次拷贝128个数字，每次之间的拷贝距离是1024B
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(accFloatRegTensor, accUbAddr, mFloatStride);

        // 累加求和
        MicroAPI::Add(addResRegTensor, accFloatRegTensor, addFloatRegTensor, castAddCopyMask);

        // 将结果拷贝到UB
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(destUbAddr, addResRegTensor, mFloatStride,
                                                                           castAddCopyMask);
    }
}

template <typename yType>
__aicore__ inline void MulScaleVf(ScaleParam<yType>& param)
{
    uint32_t mFloatStride = 256;
    uint32_t stride = 64;

    MicroAPI::RegTensor<float> accFloatRegTensor;
    MicroAPI::RegTensor<float> scaleFloatRegTensor;
    MicroAPI::RegTensor<float> mulResFloatRegTensor;
    MicroAPI::RegTensor<yType> castBf16RegTensor;
    MicroAPI::MaskReg mulCastMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg datacopyToUbMask = MicroAPI::CreateMask<yType, MicroAPI::MaskPattern::ALL>();

    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    __local_mem__ yType* destUbAddr = param.destUbAddr;
    __local_mem__ float* accUbAddr = param.accBaseAddr;

    for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
        MicroAPI::AddrReg scaleReg = MicroAPI::CreateAddrReg<float>(mIdx, 1);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(scaleFloatRegTensor, param.scaleBaseAddr, scaleReg);

        // 将累加的结果从UB拷贝到RegTensor中
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(accFloatRegTensor, accUbAddr, mFloatStride);

        MicroAPI::Mul(mulResFloatRegTensor, accFloatRegTensor, scaleFloatRegTensor, mulCastMask);

        // cast成BF16
        // x x x x x x x x-> x0 x0 x0 x0 x0 x0 x0 x0
        MicroAPI::Cast<yType, float, castTrait>(castBf16RegTensor, mulResFloatRegTensor, mulCastMask);

        // 将结果拷贝到UB
        // x0 x0 x0 x0 x0 x0 x0 x0-> x x x x x x x x
        MicroAPI::DataCopy<yType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
            destUbAddr, castBf16RegTensor, stride, datacopyToUbMask);

        // 将累加的结果从UB拷贝到RegTensor中
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(accFloatRegTensor, accUbAddr, mFloatStride);

        MicroAPI::Mul(mulResFloatRegTensor, accFloatRegTensor, scaleFloatRegTensor, mulCastMask);

        // cast成BF16
        // x x x x x x x x-> x0 x0 x0 x0 x0 x0 x0 x0
        MicroAPI::Cast<yType, float, castTrait>(castBf16RegTensor, mulResFloatRegTensor, mulCastMask);

        // 将结果拷贝到UB
        // x0 x0 x0 x0 x0 x0 x0 x0-> x x x x x x x x
        MicroAPI::DataCopy<yType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
            destUbAddr, castBf16RegTensor, stride, datacopyToUbMask);
    }
}

// 计算单个k方向的所有baseM * baseN基本块的和
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessSingleBlock(
    uint64_t kIterateIdx, uint64_t mOffset)
{
    AccParam<yType> accParam;
    accParam.mCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);

    // 如果是第0块，只做cast不做累加
    if (unlikely(kIterateIdx == 0)) {
        accParam.addBaseAddr = (__local_mem__ yType*)addPingLocalTensor_.GetPhyAddr();
        accParam.addResBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
        AscendC::VF_CALL<CastSingleBlockVf<yType>>(accParam);

        return;
    }

    if ((kIterateIdx & 1) == 0) {
        // 从bank0 加载add数据，从bank1加载acc数据，将数据结果存到bank2中
        accParam.addBaseAddr = (__local_mem__ yType*)addPingLocalTensor_.GetPhyAddr();
        accParam.accBaseAddr = (__local_mem__ float*)accPongLocalTensor_.GetPhyAddr();
        accParam.addResBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
    } else {
        // 从bank3 加载add数据，从bank2加载acc数据，将数据结果存到bank1中
        accParam.addBaseAddr = (__local_mem__ yType*)addPongLocalTensor_.GetPhyAddr();
        accParam.accBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
        accParam.addResBaseAddr = (__local_mem__ float*)accPongLocalTensor_.GetPhyAddr();
    }
    AscendC::VF_CALL<AccSingleBlockVf<yType>>(accParam);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessX1ScaleQuant(
    uint64_t mOffset, uint64_t nOffset, uint64_t kIterateIdx)
{
    // 将scale从GM搬运到UB中
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    uint64_t eight = 8;
    uint64_t calCount = CeilAlign(Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset), eight);
    DataCopy(x1ScaleUbTensor_, x1ScaleGlobal_[mOffset], calCount);

    event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);

    // 处理scale
    ScaleParam<yType> scaleParam;
    scaleParam.mCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);
    scaleParam.scaleBaseAddr = (__local_mem__ float*)x1ScaleUbTensor_.GetPhyAddr();
    scaleParam.destUbAddr = (__local_mem__ yType*)resUbTensor_.GetPhyAddr();
    if ((kIterateIdx & 1) == 0) {
        scaleParam.accBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
    } else {
        scaleParam.accBaseAddr = (__local_mem__ float*)accPongLocalTensor_.GetPhyAddr();
    }

    AscendC::VF_CALL<MulScaleVf<yType>>(scaleParam);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupFixpipeKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyResToGM(
    uint64_t mOffset, uint64_t nOffset)
{
    DataCopyParams dataCopyParams;

    uint64_t dataBlockSize = 32;
    uint64_t blockLen = CeilDiv(Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN) * 2,
                                dataBlockSize);
    dataCopyParams.blockLen = blockLen;
    dataCopyParams.blockCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);
    dataCopyParams.srcStride = 8 - blockLen;
    dataCopyParams.dstStride = CeilDiv(tiling_->matmulTiling.N * 2 - blockLen * dataBlockSize, dataBlockSize);
    uint64_t outOffset = mOffset * tiling_->matmulTiling.N + nOffset;
    DataCopy(yGlobal_[outOffset], resUbTensor_, dataCopyParams);
}
} // namespace QuantBatchMatmulV4
