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
 * \file dual_level_quant_batch_matmul_cube_compute.h
 * \brief
 */
 
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_CUBE_COMPUTE_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_CUBE_COMPUTE_H

#include "kernel_basic_intf.h"
#include "dual_level_quant_batch_matmul_block.h"
#include "dual_level_quant_batch_matmul_cube_compute_tools.h"
#include "op_kernel/math_util.h"
#include "tool_arch35.h"

using AscendC::BLOCK_CUBE;
using AscendC::Dn2NzParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace DualLevelQuantBatchMatmul::Arch35 {

#define DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename xScaleType, typename wScaleType, typename cType>

#define DLQBMM_CUBE_COMPUTE_CLASS DualLevelQuantBatchMatmulCubeCompute<xType, wType, xScaleType, wScaleType, cType>

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
class DualLevelQuantBatchMatmulCubeCompute {
public:
    __aicore__ inline DualLevelQuantBatchMatmulCubeCompute(){};
    __aicore__ inline void Init(
        __gm__ xType* x, __gm__ wType* weight, __gm__ xScaleType* xScale, __gm__ wScaleType* wScale);
    __aicore__ inline void CopyGmToL1(
        const DualLevelQbmmBasicBlockOffsetParams& blockParams, L0CopyAndCalcParams& l0Params);
    __aicore__ inline void LaunchMatmul(
        uint64_t mL1Offset, uint64_t nL1Offset, uint64_t kL1Offset, uint64_t kScaleL1Offset,
        const LocalTensor<cType>& cTmpUb, const L0CopyAndCalcParams& l0Params, const FixL0CToDstParams& fixpParams);
    __aicore__ inline void WaitMte1ToMte2(const DualLevelQbmmBasicBlockOffsetParams& blockParams);
    __aicore__ inline void SetMte1ToMte2(const DualLevelQbmmBasicBlockOffsetParams& blockParams);
    __aicore__ inline void EndSync();

private:
    __aicore__ inline void DataCopyAGmToL1(
        const DualLevelQbmmBasicBlockOffsetParams& blockParams, const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void DataCopyBGmToL1(
        const DualLevelQbmmBasicBlockOffsetParams& blockParams, const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void DataCopyScaleGmToL1(
        const DualLevelQbmmBasicBlockOffsetParams& blockParams, const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void ConfigScaleDn2NzParams(
        uint64_t rowNum, uint64_t scaleKGmGroupNum, uint64_t scaleKL1GroupNum, Dn2NzParams& dn2NzParams);

    static constexpr uint64_t C0_SIZE = GetC0Size<wType>();
    static constexpr uint64_t B8_IN_B16_NUM = 2;

    static constexpr uint64_t EVENT_MTE1_MTE2_ID = 2;
    static constexpr uint64_t EVENT_SCALE_MTE1_MTE2_ID = 4;
    static constexpr uint64_t EVENT_MTE2_MTE1_ID = 0;
    static constexpr uint64_t EVENT_M_MTE1_ID = 3;
    static constexpr uint64_t EVENT_MTE1_M_ID = 0;

    static constexpr uint32_t L0AB_BUFFER_NUM = 2;
    static constexpr uint32_t L0C_BUFFER_NUM = 3;
    static constexpr uint32_t L0AB_B4_OFFSET_ELEM = 128 * 512;
    static constexpr uint32_t L0C_B32_OFFSET_ELEM = 128 * 128;

    // L1共512KB，分完B和A/B scale之后剩余空间全部分给A
    //   0 ~ 256KB: | B_P0 (64KB)  | A_SCALE_P0 (32KB) | B_SCALE_P0 (32KB) | A_P0 (128KB) |
    // 256 ~ 512KB: | A_P1 (128KB) | B_SCALE_P1 (32KB) | A_SCALE_P1 (32KB) | B_P1 (64KB)  |
    static constexpr uint32_t A_B4_L1_BUFFER_OFFSET_ELEM = 128 * 1024 * 2;
    static constexpr uint32_t B_B4_L1_BUFFER_OFFSET_ELEM = 448 * 1024 * 2;
    static constexpr uint32_t A_SCALE_B8_L1_BUFFER_OFFSET_ELEM = 352 * 1024;
    static constexpr uint32_t B_SCALE_B8_L1_BUFFER_OFFSET_ELEM = 288 * 1024;

    LocalTensor<wType> bL1_{AscendC::TPosition::B1, 0, L1_BUFFER_SIZE_BYTE};
    LocalTensor<xScaleType> aScaleL1_{AscendC::TPosition::A1, 64 * 1024, L1_BUFFER_SIZE_BYTE - 64 * 1024};
    LocalTensor<wScaleType> bScaleL1_{AscendC::TPosition::B1, 96 * 1024, L1_BUFFER_SIZE_BYTE - 96 * 1024};
    LocalTensor<xType> aL1_{AscendC::TPosition::A1, 128 * 1024, L1_BUFFER_SIZE_BYTE - 128 * 1024};

    LocalTensor<xType> aL0_{AscendC::TPosition::A2, 0, L0A_BUFFER_SIZE_BYTE};
    LocalTensor<wType> bL0_{AscendC::TPosition::B2, 0, L0B_BUFFER_SIZE_BYTE};
    LocalTensor<cType> cL0_{AscendC::TPosition::CO1, 0, L0C_BUFFER_SIZE_BYTE};

    GlobalTensor<xType> xGm_;
    GlobalTensor<wType> weightGm_;
    GlobalTensor<half> xScaleGm_;
    GlobalTensor<half> wScaleGm_;

    uint64_t l1Mte2Idx_ = 0;
    uint64_t scaleMte2Idx_ = 0;
    uint64_t l0BufIdx_ = 0;
};

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::Init(
    __gm__ xType* x, __gm__ wType* weight, __gm__ xScaleType* xScale, __gm__ wScaleType* wScale)
{
    xGm_.SetGlobalBuffer(x);
    weightGm_.SetGlobalBuffer(weight);
    xScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(xScale));
    wScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wScale));

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_ID + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_SCALE_MTE1_MTE2_ID + i);
    }

    for (uint64_t i = 0; i < L0AB_BUFFER_NUM; i++) {
        SetFlag<HardEvent::M_MTE1>(EVENT_M_MTE1_ID + i);
    }
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::CopyGmToL1(
    const DualLevelQbmmBasicBlockOffsetParams& blockParams, L0CopyAndCalcParams& l0Params)
{
    if (blockParams.kGmOffset % blockParams.level1ScaleKL1Size == 0) {
        l0Params.scaleKL1Size = blockParams.kGmOffset + blockParams.level1ScaleKL1Size > blockParams.kSize ?
                                    blockParams.kSize - blockParams.kGmOffset :
                                    blockParams.level1ScaleKL1Size;
        DataCopyScaleGmToL1(blockParams, l0Params);
    }

    DataCopyAGmToL1(blockParams, l0Params);
    DataCopyBGmToL1(blockParams, l0Params);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_MTE2_MTE1_ID);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_MTE2_MTE1_ID);
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::DataCopyAGmToL1(
    const DualLevelQbmmBasicBlockOffsetParams& blockParams, const L0CopyAndCalcParams& l0Params)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = l0Params.mL1Size;
    nd2nzParams.dValue = l0Params.kL1Size >> 1; // DataCopy可以直接搬fp4x2，两个fp4看作一个元素，步长和数据长度等需要除2
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = blockParams.kSize >> 1;
    nd2nzParams.dstNzC0Stride = Ops::Base::CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(
        aL1_[(l1Mte2Idx_ % DOUBLE_BUFFER_NUM) * A_B4_L1_BUFFER_OFFSET_ELEM],
        xGm_[blockParams.mGmOffset * blockParams.kSize + blockParams.kGmOffset], nd2nzParams);
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::DataCopyBGmToL1(
    const DualLevelQbmmBasicBlockOffsetParams& blockParams, const L0CopyAndCalcParams& l0Params)
{
    uint64_t nAlign = Ops::Base::CeilAlign(blockParams.nSize, static_cast<uint64_t>(BLOCK_CUBE));

    // DataCopyPad不支持直接往L1搬运fp4，转换成B8搬运，步长和数据长度等需要除2
    DataCopyPad2D(
        bL1_[(l1Mte2Idx_ % DOUBLE_BUFFER_NUM) * B_B4_L1_BUFFER_OFFSET_ELEM].template ReinterpretCast<int8_t>(),
        weightGm_[blockParams.kGmOffset * nAlign + blockParams.nGmOffset * C0_SIZE].template ReinterpretCast<int8_t>(),
        Ops::Base::CeilDiv(l0Params.kL1Size, C0_SIZE), l0Params.nL1Size * C0_SIZE >> 1,
        Ops::Base::CeilAlign(l0Params.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE >> 1,
        nAlign * C0_SIZE >> 1);
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::DataCopyScaleGmToL1(
    const DualLevelQbmmBasicBlockOffsetParams& blockParams, const L0CopyAndCalcParams& l0Params)
{
    uint64_t scaleKGmGroupNum = blockParams.kSize / MX_GROUPSIZE;
    uint64_t scaleKL1GroupNum = l0Params.scaleKL1Size / MX_GROUPSIZE;

    Dn2NzParams aScaleDn2NzParams;
    ConfigScaleDn2NzParams(l0Params.mL1Size, scaleKGmGroupNum, scaleKL1GroupNum, aScaleDn2NzParams);
    uint64_t aScaleGmOffset =
        (blockParams.mGmOffset * scaleKGmGroupNum + blockParams.kGmOffset / MX_GROUPSIZE) / B8_IN_B16_NUM;
    DataCopy(
        aScaleL1_[(scaleMte2Idx_ % DOUBLE_BUFFER_NUM) * A_SCALE_B8_L1_BUFFER_OFFSET_ELEM]
            .template ReinterpretCast<half>(),
        xScaleGm_[aScaleGmOffset], aScaleDn2NzParams);

    Dn2NzParams bScaleDn2NzParams;
    ConfigScaleDn2NzParams(l0Params.nL1Size, scaleKGmGroupNum, scaleKL1GroupNum, bScaleDn2NzParams);
    uint64_t bScaleGmOffset =
        (blockParams.nGmOffset * scaleKGmGroupNum + blockParams.kGmOffset / MX_GROUPSIZE) / B8_IN_B16_NUM;
    DataCopy(
        bScaleL1_[(scaleMte2Idx_ % DOUBLE_BUFFER_NUM) * B_SCALE_B8_L1_BUFFER_OFFSET_ELEM]
            .template ReinterpretCast<half>(),
        wScaleGm_[bScaleGmOffset], bScaleDn2NzParams);
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::ConfigScaleDn2NzParams(
    uint64_t rowNum, uint64_t scaleKGmGroupNum, uint64_t scaleKL1GroupNum, Dn2NzParams& dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum; // 矩阵的行数，即待搬运的mxScaleA的m或mxScaleB的n
    dn2NzParams.nValue = scaleKL1GroupNum / B8_IN_B16_NUM; // 矩阵的列数，使用B16搬B8需要除以2
    dn2NzParams.srcDnMatrixStride = 0;
    dn2NzParams.srcDValue = scaleKGmGroupNum / B8_IN_B16_NUM; // 源矩阵一行所含B16元素个数
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzC0Stride = scaleKL1GroupNum / B8_IN_B16_NUM;
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = 0;
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::LaunchMatmul(
    uint64_t mL1Offset, uint64_t nL1Offset, uint64_t kL1Offset, uint64_t kScaleL1Offset,
    const LocalTensor<cType>& cTmpUb, const L0CopyAndCalcParams& l0Params, const FixL0CToDstParams& fixpParams)
{
    LocalTensor<xType> aL0Tensor = aL0_[(l0BufIdx_ % L0AB_BUFFER_NUM) * L0AB_B4_OFFSET_ELEM];
    LocalTensor<wType> bL0Tensor = bL0_[(l0BufIdx_ % L0AB_BUFFER_NUM) * L0AB_B4_OFFSET_ELEM];
    LocalTensor<cType> cL0Tensor = cL0_[(l0BufIdx_ % L0C_BUFFER_NUM) * L0C_B32_OFFSET_ELEM];
    WaitFlag<HardEvent::M_MTE1>(EVENT_M_MTE1_ID + l0BufIdx_ % L0AB_BUFFER_NUM);
    LoadAAndScaleL1ToL0<xType, xType, xScaleType>(
        aL0Tensor,
        aL1_
            [(l1Mte2Idx_ % DOUBLE_BUFFER_NUM) * A_B4_L1_BUFFER_OFFSET_ELEM + mL1Offset * C0_SIZE +
             kL1Offset * Ops::Base::CeilAlign(l0Params.mL1Size, static_cast<uint64_t>(BLOCK_CUBE))],
        aScaleL1_
            [(scaleMte2Idx_ % DOUBLE_BUFFER_NUM) * A_SCALE_B8_L1_BUFFER_OFFSET_ELEM +
             mL1Offset * l0Params.scaleKL1Size / MX_GROUPSIZE + kScaleL1Offset / MX_GROUPSIZE * BLOCK_CUBE],
        l0Params);
    LoadBAndScaleL1ToL0<wType, wType, wScaleType>(
        bL0Tensor,
        bL1_
            [(l1Mte2Idx_ % DOUBLE_BUFFER_NUM) * B_B4_L1_BUFFER_OFFSET_ELEM + nL1Offset * C0_SIZE +
             kL1Offset * Ops::Base::CeilAlign(l0Params.nL1Size, static_cast<uint64_t>(BLOCK_CUBE))],
        bScaleL1_
            [(scaleMte2Idx_ % DOUBLE_BUFFER_NUM) * B_SCALE_B8_L1_BUFFER_OFFSET_ELEM +
             nL1Offset * l0Params.scaleKL1Size / MX_GROUPSIZE + kScaleL1Offset / MX_GROUPSIZE * BLOCK_CUBE],
        l0Params);
    SetFlag<HardEvent::MTE1_M>(EVENT_MTE1_M_ID);
    WaitFlag<HardEvent::MTE1_M>(EVENT_MTE1_M_ID);
    MmadCompute<cType, xType, wType>(cL0Tensor, aL0Tensor, bL0Tensor, l0Params);
    SetFlag<HardEvent::M_MTE1>(EVENT_M_MTE1_ID + l0BufIdx_ % L0AB_BUFFER_NUM);
    FixL0CToDst<cType, cType>(cTmpUb, cL0Tensor, fixpParams);
    l0BufIdx_++;
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::WaitMte1ToMte2(const DualLevelQbmmBasicBlockOffsetParams& blockParams)
{
    if (blockParams.kGmOffset % blockParams.level1ScaleKL1Size == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_SCALE_MTE1_MTE2_ID + (scaleMte2Idx_ % DOUBLE_BUFFER_NUM));
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_ID + (l1Mte2Idx_ % DOUBLE_BUFFER_NUM));
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::SetMte1ToMte2(const DualLevelQbmmBasicBlockOffsetParams& blockParams)
{
    if ((blockParams.kGmOffset + blockParams.l0BaseK) % blockParams.level1ScaleKL1Size == 0 ||
        blockParams.kGmOffset + blockParams.l0BaseK >= blockParams.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_SCALE_MTE1_MTE2_ID + (scaleMte2Idx_ % DOUBLE_BUFFER_NUM));
        scaleMte2Idx_++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_ID + (l1Mte2Idx_ % DOUBLE_BUFFER_NUM));
    l1Mte2Idx_++;
}

DLQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_CUBE_COMPUTE_CLASS::EndSync()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_ID + i);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_SCALE_MTE1_MTE2_ID + i);
    }

    for (uint64_t i = 0; i < L0AB_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_M_MTE1_ID + i);
    }
}

} // namespace DualLevelQuantBatchMatmul::Arch35
#endif