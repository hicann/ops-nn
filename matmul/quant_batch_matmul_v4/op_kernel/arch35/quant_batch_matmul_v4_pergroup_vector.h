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
 * \file quant_batch_matmul_v4_pergroup_vector.h
 * \brief
 */

#pragma once

#include "../../common/arch35/weight_quant_batch_matmul_cube_compute_tools.h"
#include "tool.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS \
    template <typename xType, typename wType, typename yType, bool bTrans, typename x1ScaleType, typename x2ScaleType>

#define LOCAL_TEMPLATE_FUNC_PARAMS xType, wType, yType, bTrans, x1ScaleType, x2ScaleType

namespace QuantBatchMatmulV4 {

template <typename yType, typename x2ScaleType>
struct AccParamV2 {
    uint64_t mCount;
    uint64_t nCount;
    __local_mem__ yType* addBaseAddr;
    __local_mem__ float* accBaseAddr;
    __local_mem__ float* addResBaseAddr;
    __local_mem__ x2ScaleType* x2ScaleBaseAddr;
};

template <typename yType>
struct ScaleParamV2 {
    uint64_t mCount;
    uint64_t nCount;
    uint64_t nSize;

    __local_mem__ float* scaleBaseAddr;
    __local_mem__ yType* destUbAddr;
    __local_mem__ float* accBaseAddr;
};

LOCAL_TEMPLATE_CLASS_PARAMS
class QuantBatchMatmulV4PerGroupVectorKernel {
public:
    __aicore__ inline QuantBatchMatmulV4PerGroupVectorKernel(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y,
                                GM_ADDR workspace,
                                const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tilingData);

    __aicore__ inline void InitUbBuffs();
    __aicore__ inline void ProcessSingleBlock(uint64_t kIterateIdx, uint64_t mOffset);
    __aicore__ inline void CopyResToGM(uint64_t mOffset, uint64_t nOffset);

    __aicore__ inline void InitInputs(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale,
                                      GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitCubeBuffs();

    __aicore__ inline void GetABToL1(uint64_t mOffset, uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyX1Nd2Nz(uint64_t mOffset, uint64_t kOffset);
    __aicore__ inline void CopyX2Gm2L1(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyX2ScaleGm2UB(uint64_t nOffset, uint64_t kOffset, uint64_t kIterateIdx);

    __aicore__ inline void IterateMatmul(uint64_t kInnerIdx);

    __aicore__ inline void UpdateMmParams(uint64_t mOffset, uint64_t nOffset);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBaseBlockCompute(uint64_t mOffset, uint64_t nOffset, uint64_t totalKBlocks);
    __aicore__ inline void ProcessX1ScaleQuant(uint64_t mOffset, uint64_t nOffset);

public:
    using inputWType = MatmulType<TPosition::TSCM, CubeFormat::NZ, DTYPE_X2, bTrans, LayoutMode::NONE, false,
                                  TPosition::GM>;
    const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tiling_;

protected:
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr int64_t L0C_BUFFER_SIZE = 262144;
    static constexpr int64_t L0A_BUFFER_SIZE = 65536;
    static constexpr int64_t L0B_BUFFER_SIZE = 65536;
    static constexpr uint8_t DOUBLE_BUFFER = 2;
    static constexpr int64_t BUFFER_HALF_SHL = 15;
    // S8S4 pergroup，Vector量化，输出为fp32，写入UB时按1024 B跳写
    static constexpr int64_t FIXPIPE_TO_UB_DSTSTRIDE_256 = 256;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_2 = 2;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_3 = 3;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_4 = 4;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_5 = 5;
    static constexpr uint8_t MTE_SYNC_DB_FLAG_IDS_1[DOUBLE_BUFFER] = {MTE_SYNC_FLAG_ID_4, MTE_SYNC_FLAG_ID_5};

    int32_t curBlockIdx_;

    uint64_t curL1BufIdx_ = 0;
    uint64_t curL0BufIdx_ = 0;
    uint64_t curScaleBufIdx_ = 0; // 记录x2scale的pingpong
    // fixPipe到UB的buff index，开double buffer
    uint64_t curFixOutUbBufIdx_ = 0;
    uint64_t computeIterIdx_ = 0;
    uint64_t singleCoreKCount_ = 0;

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

    // FixPipe等Vector的核间同步总次数
    uint64_t pipeVToFixSyncCount_;

    // 放在bank0的fixpipe结果
    LocalTensor<int32_t> addPingLocalTensor_;
    // 放在bank1的累加和结果
    LocalTensor<float> accPongLocalTensor_;
    // 放在bank2的累加和结果
    LocalTensor<float> accPingLocalTensor_;
    // 放在bank3的fixpipe结果
    LocalTensor<int32_t> addPongLocalTensor_;

    LocalTensor<float> x1ScaleUbTensor_;
    LocalTensor<yType> resUbTensor_;
    LocalTensor<x2ScaleType> x2ScalePingTensor_;
    LocalTensor<x2ScaleType> x2ScalePongTensor_;

    LocalTensor<int32_t> cL0Tensor_;
    LocalTensor<xType> aL0Tensor_;
    LocalTensor<wType> bL0Tensor_;

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
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace,
    const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tilingData)
{
    tiling_ = tilingData;

    curBlockIdx_ = GetBlockIdx();

    kBlockSize_ = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;

    InitInputs(x1, x2, x1_scale, x2_scale, y_scale, y, workspace);
    InitUbBuffs();

    // 申请AL1和BL1 MTE1_MTE2同步事件ID
    if ASCEND_IS_AIC {
        InitCubeBuffs();
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[1]);
    } else {
        SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
        SetFlag<HardEvent::MTE3_V>(MTE_SYNC_FLAG_ID_2);
        SetFlag<HardEvent::V_MTE2>(MTE_SYNC_FLAG_ID_2);
    }

    singleCoreKCount_ = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, tiling_->matmulTiling.baseK);
    uint64_t mLoopNum = CeilDiv<uint64_t>(CeilDiv<uint64_t>(tiling_->matmulTiling.M, tiling_->cubeBlockDimM),
                                          tiling_->matmulTiling.baseM);
    uint64_t nLoopNum = CeilDiv<uint64_t>(CeilDiv<uint64_t>(tiling_->matmulTiling.N, tiling_->cubeBlockDimN),
                                          tiling_->matmulTiling.baseN);
    uint64_t totalInnerLoopNum = singleCoreKCount_ * mLoopNum * nLoopNum;
    pipeVToFixSyncCount_ = totalInnerLoopNum > 2 ? totalInnerLoopNum - 2 : 0; // 2为double buff，最后两次不用设置同步
}

/*
 *  功能：申请各输入的 global buffer
 */
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitInputs(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace)
{
    // 申请 Global Buffer
    if ASCEND_IS_AIC {
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
        wGlobalU8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x2));
    }

    if ASCEND_IS_AIV {
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
        x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x1ScaleType*>(x1_scale));
        x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x2ScaleType*>(x2_scale));
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitCubeBuffs()
{
    // B矩阵L1 Buff大小，BL1是int4类型
    uint64_t bL1DataSizes = tiling_->matmulTiling.baseN * kBlockSize_;
    uint64_t bL1DataBytes = bL1DataSizes >> 1;

    // A矩阵L1 Buff大小
    uint64_t aL1DataSizes = tiling_->matmulTiling.baseM * kBlockSize_;
    uint64_t aL1DataBytes = aL1DataSizes * sizeof(xType);

    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    // L1 ping buff分配
    bL1PingLocalBuf_ = AscendC::LocalTensor<wType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
    l1BufOffsetPart1 += bL1DataBytes;

    aL1PingLocalBuf_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, aL1DataSizes);
    l1BufOffsetPart1 += aL1DataBytes;

    // L1 pong buff分配
    bL1PongLocalBuf_ = AscendC::LocalTensor<wType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
    l1BufOffsetPart2 += bL1DataBytes;

    aL1PongLocalBuf_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, aL1DataSizes);
    l1BufOffsetPart2 += aL1DataBytes;

    cL0Tensor_ = AscendC::LocalTensor<int32_t>(TPosition::CO1, 0, L0C_BUFFER_SIZE / sizeof(int32_t));
    aL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::A2, 0, L0A_BUFFER_SIZE / sizeof(xType));
    bL0Tensor_ = AscendC::LocalTensor<wType>(TPosition::B2, 0, L0B_BUFFER_SIZE << 1);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    uint64_t usedCoreNum = tiling_->matmulTiling.usedCoreNum;
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
        WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
        WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[1]);
    } else {
        WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
        WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_FLAG_ID_2);
        WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_FLAG_ID_2);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateMmParams(
    uint64_t mOffset, uint64_t nOffset)
{
    curMSize_ = Min<uint64_t>(tiling_->matmulTiling.M - mOffset, tiling_->matmulTiling.baseM);
    curNSize_ = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
    if ASCEND_IS_AIC {
        curMAlignSize_ = CeilAlign<uint64_t>(curMSize_, BLOCK_CUBE);
        curNAlignSize_ = CeilAlign<uint64_t>(curNSize_, BLOCK_CUBE);

        aL1BaseOffset_ = tiling_->matmulTiling.baseK * curMAlignSize_;
        bL1BaseOffset_ = tiling_->matmulTiling.baseK * 64; // int4 n0 size为64
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessBaseBlockCompute(
    uint64_t mOffset, uint64_t nOffset, uint64_t totalKBlocks)
{
    uint64_t kIterateIdx = 0;
    for (uint64_t kOffset = 0; kOffset < tiling_->matmulTiling.Ka; kOffset += kBlockSize_) {
        // A、B矩阵K方向搬入 L1 Size 计算
        curKL1Size_ = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
        if ASCEND_IS_AIC {
            GetABToL1(mOffset, nOffset, kOffset);

            // 等待GM搬入L1完成后启动Matmul计算
            SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_FLAG_ID_2);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_FLAG_ID_2);
        } else {
            if (kIterateIdx % (tiling_->matmulTiling.stepKa * tiling_->scaleFactor) == 0) {
                CopyX2ScaleGm2UB(nOffset, kOffset, kIterateIdx);
            }
        }

        // k内部循环，因为需要在Vector累加，所以需要保证每次Matmul的K size在同一个group内
        uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKL1Size_, tiling_->matmulTiling.baseK);
        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            if ASCEND_IS_AIC {
                // FixPipe交替写入UB的两块buffer，前两次不用等Vector同步
                if (likely(computeIterIdx_ > 1)) {
                    WaitForVector();
                }

                IterateMatmul(kInnerIdx);
                NotifyVector();
            }

            if ASCEND_IS_AIV {
                WaitForCube();
                ProcessSingleBlock(kIterateIdx, mOffset);
                // FixPipe交替写入UB的两块buffer，最后两次baseK不需要设置核间同步标记防止不匹配
                if (likely(computeIterIdx_ < pipeVToFixSyncCount_)) {
                    NotifyCube();
                }
            }
            kIterateIdx++;
            computeIterIdx_++;
        }

        if ASCEND_IS_AIC {
            curL1BufIdx_ == IDX_0 ? SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]) :
                                    SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
            // 一次K方向计算结束，A和B矩阵切换对应的L1 Buff
            curL1BufIdx_ = curL1BufIdx_ ^ 1;
        } else {
            if (kIterateIdx % (tiling_->matmulTiling.stepKa * tiling_->scaleFactor) == 0 ||
                kOffset + kBlockSize_ >= tiling_->matmulTiling.Ka) {
                curScaleBufIdx_ == IDX_0 ? SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]) :
                                           SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
                curScaleBufIdx_ = curScaleBufIdx_ ^ 1;
            }
        }
    }

    if ASCEND_IS_AIV {
        // 基本块累加结束后，做x1_scale的量化
        ProcessX1ScaleQuant(mOffset, nOffset);

        SetFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_2);
        WaitFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_2);
        // 将结果搬运到GM
        CopyResToGM(mOffset, nOffset);
        SetFlag<HardEvent::MTE3_V>(MTE_SYNC_FLAG_ID_2);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetABToL1(uint64_t mOffset,
                                                                                                     uint64_t nOffset,
                                                                                                     uint64_t kOffset)
{
    // 等待L1 buffer可用
    curL1BufIdx_ == IDX_0 ? WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]) :
                            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);

    CopyX1Nd2Nz(mOffset, kOffset);

    // A、B共用1组同步，拷贝A前等待L1 buffer可用
    CopyX2Gm2L1(nOffset, kOffset);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX1Nd2Nz(uint64_t mOffset,
                                                                                                       uint64_t kOffset)
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
    DataCopy((curL1BufIdx_ == IDX_0) ? aL1PingLocalBuf_ : aL1PongLocalBuf_, xGlobal_[aGmOffset], nd2nzParams,
             tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX2Gm2L1(uint64_t nOffset,
                                                                                                       uint64_t kOffset)
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
                      wGlobalU8_[x2GmOffset], dataCopyParams, tiling_->enablePreAlloc);
}

/**
 * @brief 将 X2 Scale 数据从 GM 拷贝到 L1，每次拷贝 nL1Size * stepK 的大小供 FixPipe 使用。
 */
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX2ScaleGm2UB(
    uint64_t nOffset, uint64_t kOffset, uint64_t kIterateIdx)
{
    uint64_t bScaleGmOffset = kIterateIdx * tiling_->matmulTiling.N + nOffset;

    uint64_t scaleKSize = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_ * tiling_->scaleFactor);
    uint16_t blockCount = static_cast<uint16_t>(CeilDiv<uint64_t>(scaleKSize, tiling_->matmulTiling.baseK));

    // srcStride:(N - nOffSet - nL1Size) + nOffset = (N - nL1Size)
    uint16_t srcStride = static_cast<uint16_t>(((tiling_->matmulTiling.N - curNSize_) * sizeof(x2ScaleType)) /
                                               32); // 单位：32B
    AscendC::DataCopyParams intriParams;
    intriParams.blockCount = blockCount;
    intriParams.blockLen = static_cast<uint16_t>(curNSize_ * sizeof(x2ScaleType) / ONE_BLK_SIZE);
    intriParams.srcStride = srcStride;
    intriParams.dstStride = 0;

    (curScaleBufIdx_ == IDX_0) ? WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]) :
                                 WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
    DataCopy((curScaleBufIdx_ == IDX_0) ? x2ScalePingTensor_ : x2ScalePongTensor_, x2ScaleGlobal_[bScaleGmOffset],
             intriParams, tiling_->enablePreAlloc);

    SetFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_2);
    WaitFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_2);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IterateMatmul(
    uint64_t kInnerIdx)
{
    WeightQuantBatchMatmul::Arch35::L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL0Size = curMSize_;
    l0CopyAndCalcParams.nL0Size = curNSize_;
    l0CopyAndCalcParams.nL1Size = l0CopyAndCalcParams.nL0Size;
    l0CopyAndCalcParams.mL1Size = l0CopyAndCalcParams.mL0Size;
    l0CopyAndCalcParams.kL1Size = curKL1Size_;
    // per-group baseK=group_size，tiling保证K方向不会带尾块，这里直接设置成baseK就可以
    l0CopyAndCalcParams.kL0Size = tiling_->matmulTiling.baseK;
    l0CopyAndCalcParams.isFirstKLoop = true;
    l0CopyAndCalcParams.isLastKLoop = true;
    WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[curL0BufIdx_]);
    uint64_t aL1Offset = kInnerIdx * aL1BaseOffset_;
    LocalTensor<xType> aL0Tensor = aL0Tensor_[curL0BufIdx_ << BUFFER_HALF_SHL];
    WeightQuantBatchMatmul::Arch35::LoadAL1ToL0<xType, xType>(
        aL0Tensor, (curL1BufIdx_ == IDX_0) ? aL1PingLocalBuf_[aL1Offset] : aL1PongLocalBuf_[aL1Offset],
        l0CopyAndCalcParams);
    // kOffSet * n0_size
    uint64_t bL1Offset = kInnerIdx * bL1BaseOffset_;
    LocalTensor<wType> bL0Tensor = bL0Tensor_[curL0BufIdx_ << (BUFFER_HALF_SHL + 1)];
    WeightQuantBatchMatmul::Arch35::LoadBL1ToL0<wType, wType, false>(
        bL0Tensor, (curL1BufIdx_ == IDX_0) ? bL1PingLocalBuf_[bL1Offset] : bL1PongLocalBuf_[bL1Offset],
        l0CopyAndCalcParams);
    SetFlag<HardEvent::MTE1_M>(MTE_SYNC_DB_FLAG_IDS_1[curL0BufIdx_]);

    LocalTensor<int32_t> cL0Tensor = cL0Tensor_[curL0BufIdx_ << BUFFER_HALF_SHL];
    WaitFlag<HardEvent::MTE1_M>(MTE_SYNC_DB_FLAG_IDS_1[curL0BufIdx_]);

    WeightQuantBatchMatmul::Arch35::MmadCompute<int32_t, xType, wType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                       l0CopyAndCalcParams);
    SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[curL0BufIdx_]);
    curL0BufIdx_ ^= 1;

    WeightQuantBatchMatmul::Arch35::FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = curMSize_;
    fixL0CToDstParams.nL0Size = curNSize_;
    fixL0CToDstParams.outNSize = FIXPIPE_TO_UB_DSTSTRIDE_256;
    fixL0CToDstParams.splitNSize = 64;
    fixL0CToDstParams.dstNdStride = curMSize_ * FIXPIPE_TO_UB_DSTSTRIDE_256;
    // Fixpipe结果需要交替写入UB
    WeightQuantBatchMatmul::Arch35::FixL0CToDst<int32_t, int32_t, true>(
        (curFixOutUbBufIdx_ == 0 ? addPingLocalTensor_ : addPongLocalTensor_), cL0Tensor, fixL0CToDstParams);

    curFixOutUbBufIdx_ ^= 1;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitUbBuffs()
{
    // 第一个4指的是4个buffer，第二个4指的是buffer中的数据类型是float或int32，每个数是4个字节
    uint32_t bankUbByte = 4 * tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 4;
    addPingLocalTensor_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECIN, 0, bankUbByte / sizeof(int32_t));
    accPongLocalTensor_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, 256,
                                                      (bankUbByte - 256) / sizeof(float));
    accPingLocalTensor_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, 512,
                                                      (bankUbByte - 512) / sizeof(float));
    addPongLocalTensor_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECIN, 768,
                                                        (bankUbByte - 768) / sizeof(int32_t));

    uint64_t ubBufOffset = bankUbByte;
    uint64_t antiScaleUbSize = tiling_->scaleFactor * tiling_->matmulTiling.stepKb * tiling_->matmulTiling.baseN;
    x2ScalePingTensor_ = AscendC::LocalTensor<x2ScaleType>(AscendC::TPosition::VECIN, ubBufOffset, antiScaleUbSize);
    ubBufOffset += antiScaleUbSize * sizeof(x2ScaleType);
    x2ScalePongTensor_ = AscendC::LocalTensor<x2ScaleType>(AscendC::TPosition::VECIN, ubBufOffset, antiScaleUbSize);
    ubBufOffset += antiScaleUbSize * sizeof(x2ScaleType);

    x1ScaleUbTensor_ = AscendC::LocalTensor<x1ScaleType>(AscendC::TPosition::VECIN, ubBufOffset,
                                                         tiling_->matmulTiling.baseM);
    ubBufOffset += tiling_->matmulTiling.baseM * sizeof(x1ScaleType);

    resUbTensor_ = AscendC::LocalTensor<yType>(AscendC::TPosition::VECOUT, ubBufOffset,
                                               tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN);
}

// cast单个block并且乘scale搬运到目标地址
template <typename yType, typename x2ScaleType>
__aicore__ inline void CastSingleBlockVf(const AccParamV2<yType, x2ScaleType>& param)
{
    uint32_t nBfStride = 64;
    uint32_t b32BankStride = 256;
    MicroAPI::RegTensor<yType> addS32RegTensor;
    MicroAPI::RegTensor<float> addF32RegTensor;
    MicroAPI::RegTensor<float> mulF32RegTensor;
    MicroAPI::RegTensor<x2ScaleType> x2ScaleRegTensor;

    MicroAPI::MaskReg castMulCopyMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait castTraitNorm = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_ROUND};
    __local_mem__ float* destUbAddr = param.addResBaseAddr;
    __local_mem__ yType* addUbAddr = param.addBaseAddr;
    __local_mem__ x2ScaleType* x2ScaleUbAddr = param.x2ScaleBaseAddr;

    if (param.mCount == 1) {
#pragma unroll 2
        for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
            MicroAPI::AddrReg addInAddrReg = MicroAPI::CreateAddrReg<int32_t>(nIdx, b32BankStride);
            MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(nIdx, b32BankStride);
            MicroAPI::AddrReg scaleInAddrReg = MicroAPI::CreateAddrReg<x2ScaleType>(nIdx, nBfStride);
            MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_NORM>(addS32RegTensor, addUbAddr, addInAddrReg);
            MicroAPI::DataCopy<x2ScaleType, MicroAPI::LoadDist::DIST_NORM>(x2ScaleRegTensor, x2ScaleUbAddr,
                                                                           scaleInAddrReg);
            MicroAPI::Cast<float, yType, castTraitNorm>(addF32RegTensor, addS32RegTensor, castMulCopyMask);
            MicroAPI::Mul(mulF32RegTensor, addF32RegTensor, x2ScaleRegTensor, castMulCopyMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, mulF32RegTensor, outAddrReg,
                                                                          castMulCopyMask);
        }
    } else {
        for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
            MicroAPI::AddrReg scaleInAddrReg = MicroAPI::CreateAddrReg<x2ScaleType>(nIdx, nBfStride);
            MicroAPI::DataCopy<x2ScaleType, MicroAPI::LoadDist::DIST_NORM>(x2ScaleRegTensor, x2ScaleUbAddr,
                                                                           scaleInAddrReg);
#pragma unroll 4
            for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
                MicroAPI::AddrReg addInAddrReg = MicroAPI::CreateAddrReg<int32_t>(nIdx, param.mCount * b32BankStride,
                                                                                  mIdx, b32BankStride);
                MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(nIdx, b32BankStride, mIdx,
                                                                              param.nCount * b32BankStride);
                MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_NORM>(addS32RegTensor, addUbAddr, addInAddrReg);
                MicroAPI::Cast<float, yType, castTraitNorm>(addF32RegTensor, addS32RegTensor, castMulCopyMask);
                MicroAPI::Mul(mulF32RegTensor, addF32RegTensor, x2ScaleRegTensor, castMulCopyMask);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, mulF32RegTensor, outAddrReg,
                                                                              castMulCopyMask);
            }
        }
    }
}

template <typename yType, typename x2ScaleType>
__aicore__ inline void AccSingleBlockVf(const AccParamV2<yType, x2ScaleType>& param)
{
    uint32_t nBfStride = 64;
    uint32_t b32BankStride = 256;
    MicroAPI::RegTensor<yType> addS32RegTensor;
    MicroAPI::RegTensor<float> addF32RegTensor;
    MicroAPI::RegTensor<float> accF32RegTensor;
    MicroAPI::RegTensor<float> addResF32RegTensor;
    MicroAPI::RegTensor<float> mulF32RegTensor;
    MicroAPI::RegTensor<x2ScaleType> x2ScaleRegTensor;

    MicroAPI::MaskReg castMulCopyMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait castTraitNorm = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_ROUND};
    __local_mem__ float* destUbAddr = param.addResBaseAddr;
    __local_mem__ float* accUbAddr = param.accBaseAddr;
    __local_mem__ yType* addUbAddr = param.addBaseAddr;
    __local_mem__ x2ScaleType* x2ScaleUbAddr = param.x2ScaleBaseAddr;

    if (param.mCount == 1) {
#pragma unroll 2
        for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
            MicroAPI::AddrReg addInAddrReg = MicroAPI::CreateAddrReg<int32_t>(nIdx, b32BankStride);
            MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(nIdx, b32BankStride);
            MicroAPI::AddrReg scaleInAddrReg = MicroAPI::CreateAddrReg<x2ScaleType>(nIdx, nBfStride);
            MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_NORM>(addS32RegTensor, addUbAddr, addInAddrReg);
            MicroAPI::DataCopy<x2ScaleType, MicroAPI::LoadDist::DIST_NORM>(x2ScaleRegTensor, x2ScaleUbAddr,
                                                                           scaleInAddrReg);
            MicroAPI::Cast<float, yType, castTraitNorm>(addF32RegTensor, addS32RegTensor, castMulCopyMask);
            MicroAPI::Mul(mulF32RegTensor, addF32RegTensor, x2ScaleRegTensor, castMulCopyMask);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accF32RegTensor, accUbAddr, outAddrReg);
            MicroAPI::Add(addResF32RegTensor, mulF32RegTensor, accF32RegTensor, castMulCopyMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, addResF32RegTensor, outAddrReg,
                                                                          castMulCopyMask);
        }
    } else {
        for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
            MicroAPI::AddrReg scaleInAddrReg = MicroAPI::CreateAddrReg<x2ScaleType>(nIdx, nBfStride);
            MicroAPI::DataCopy<x2ScaleType, MicroAPI::LoadDist::DIST_NORM>(x2ScaleRegTensor, x2ScaleUbAddr,
                                                                           scaleInAddrReg);
#pragma unroll 4
            for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
                MicroAPI::AddrReg addInAddrReg = MicroAPI::CreateAddrReg<int32_t>(nIdx, param.mCount * b32BankStride,
                                                                                  mIdx, b32BankStride);
                MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(nIdx, b32BankStride, mIdx,
                                                                              param.nCount * b32BankStride);
                MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_NORM>(addS32RegTensor, addUbAddr, addInAddrReg);
                MicroAPI::Cast<float, yType, castTraitNorm>(addF32RegTensor, addS32RegTensor, castMulCopyMask);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accF32RegTensor, accUbAddr, outAddrReg);
                MicroAPI::MulAddDst<float>(accF32RegTensor, addF32RegTensor, x2ScaleRegTensor, castMulCopyMask);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, accF32RegTensor, outAddrReg,
                                                                              castMulCopyMask);
            }
        }
    }
}

template <typename yType>
__aicore__ inline void MulScaleVfV2(ScaleParamV2<yType>& param)
{
    uint32_t b32BankStride = 256;
    uint32_t stride = 64;

    MicroAPI::RegTensor<float> accF32RegTensor;
    MicroAPI::RegTensor<float> scaleF32RegTensor;
    MicroAPI::RegTensor<float> mulResF32RegTensor;
    MicroAPI::RegTensor<yType> castF16RegTensor;
    MicroAPI::MaskReg mulCastMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg datacopyToUbMask = MicroAPI::CreateMask<yType, MicroAPI::MaskPattern::ALL>();

    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    __local_mem__ yType* destUbAddr = param.destUbAddr;
    __local_mem__ float* accUbAddr = param.accBaseAddr;

    for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
        MicroAPI::AddrReg scaleReg = MicroAPI::CreateAddrReg<float>(mIdx, 1);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(scaleF32RegTensor, param.scaleBaseAddr, scaleReg);
#pragma unroll 2
        for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
            MicroAPI::AddrReg accAddrReg = MicroAPI::CreateAddrReg<float>(mIdx, param.nCount * b32BankStride, nIdx,
                                                                          b32BankStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accF32RegTensor, accUbAddr, accAddrReg);
            MicroAPI::Mul(mulResF32RegTensor, accF32RegTensor, scaleF32RegTensor, mulCastMask);
            MicroAPI::Cast<yType, float, castTrait>(castF16RegTensor, mulResF32RegTensor, mulCastMask);
            MicroAPI::AddrReg dstAddrReg = MicroAPI::CreateAddrReg<yType>(mIdx, param.nSize, nIdx, stride);
            MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_PACK_B32>(destUbAddr, castF16RegTensor, dstAddrReg,
                                                                          datacopyToUbMask);
        }
    }
}

// 计算单个k方向的所有baseM * baseN基本块的和
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessSingleBlock(
    uint64_t kIterateIdx, uint64_t mOffset)
{
    AccParamV2<int32_t, x2ScaleType> accParam;
    accParam.mCount = curMSize_;
    accParam.nCount = CeilDiv<uint64_t>(curNSize_, 64);

    if ((computeIterIdx_ & 1) == 0) {
        // 从bank0 加载add数据，从bank1加载acc数据，将数据结果存到bank2中
        accParam.addBaseAddr = (__local_mem__ int32_t*)addPingLocalTensor_.GetPhyAddr();
        accParam.accBaseAddr = (__local_mem__ float*)accPongLocalTensor_.GetPhyAddr();
        accParam.addResBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
    } else {
        // 从bank3 加载add数据，从bank2加载acc数据，将数据结果存到bank1中
        accParam.addBaseAddr = (__local_mem__ int32_t*)addPongLocalTensor_.GetPhyAddr();
        accParam.accBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
        accParam.addResBaseAddr = (__local_mem__ float*)accPongLocalTensor_.GetPhyAddr();
    }
    if ((curScaleBufIdx_ & 1) == 0) {
        accParam.x2ScaleBaseAddr = (__local_mem__ x2ScaleType*)
                                       x2ScalePingTensor_[(kIterateIdx %
                                                           (tiling_->matmulTiling.stepKa * tiling_->scaleFactor)) *
                                                          curNSize_]
                                           .GetPhyAddr();
    } else {
        accParam.x2ScaleBaseAddr = (__local_mem__ x2ScaleType*)
                                       x2ScalePongTensor_[(kIterateIdx %
                                                           (tiling_->matmulTiling.stepKa * tiling_->scaleFactor)) *
                                                          curNSize_]
                                           .GetPhyAddr();
    }

    if (unlikely(kIterateIdx == 0)) { // 如果是第0块，只做cast不做累加
        AscendC::VF_CALL<CastSingleBlockVf<int32_t, x2ScaleType>>(accParam);
    } else {
        AscendC::VF_CALL<AccSingleBlockVf<int32_t, x2ScaleType>>(accParam);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessX1ScaleQuant(
    uint64_t mOffset, uint64_t nOffset)
{
    // 将scale从GM搬运到UB中
    WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_FLAG_ID_2);

    uint64_t eight = 8;
    uint64_t calCount = CeilAlign(Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset), eight);
    AscendC::DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = static_cast<uint16_t>(calCount * sizeof(x1ScaleType) / ONE_BLK_SIZE);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    DataCopy(x1ScaleUbTensor_, x1ScaleGlobal_[mOffset], intriParams, true);

    SetFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_3);
    WaitFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_3);
    WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_FLAG_ID_2);
    // 处理scale
    ScaleParamV2<yType> scaleParam;
    scaleParam.mCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);
    scaleParam.nCount = CeilDiv<uint64_t>(curNSize_, 64);
    scaleParam.nSize = curNSize_;
    scaleParam.scaleBaseAddr = (__local_mem__ float*)x1ScaleUbTensor_.GetPhyAddr();
    scaleParam.destUbAddr = (__local_mem__ yType*)resUbTensor_.GetPhyAddr();
    if ((computeIterIdx_ & 1) != 0) {
        scaleParam.accBaseAddr = (__local_mem__ float*)accPingLocalTensor_.GetPhyAddr();
    } else {
        scaleParam.accBaseAddr = (__local_mem__ float*)accPongLocalTensor_.GetPhyAddr();
    }

    AscendC::VF_CALL<MulScaleVfV2<yType>>(scaleParam);
    SetFlag<HardEvent::V_MTE2>(MTE_SYNC_FLAG_ID_2);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4PerGroupVectorKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyResToGM(uint64_t mOffset,
                                                                                                       uint64_t nOffset)
{
    DataCopyParams dataCopyParams;

    uint64_t dataBlockSize = 32;
    uint64_t blockLen = CeilDiv(Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN) * 2,
                                dataBlockSize);
    dataCopyParams.blockLen = blockLen;
    dataCopyParams.blockCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = CeilDiv(tiling_->matmulTiling.N * 2 - blockLen * dataBlockSize, dataBlockSize);
    uint64_t outOffset = mOffset * tiling_->matmulTiling.N + nOffset;
    DataCopy(yGlobal_[outOffset], resUbTensor_, dataCopyParams);
}
} // namespace QuantBatchMatmulV4
