/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_experiment_cube_compute.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_CUBE_COMPUTE_H
#define WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_CUBE_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_experiment_tool.h"

using AscendC::AIC;
using AscendC::BLOCK_CUBE;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace WeightQuantBatchMatmulExperimental {

#define WQBMM_EXP_CUBE_COMPUTE_TEMPLATE_PARAM template <typename xType, typename wType, typename CUBE_FIXP_DTYPE>

#define WQBMM_EXP_CUBE_COMPUTE_CLASS WqbmmExpCubeCompute<xType, wType, CUBE_FIXP_DTYPE>

WQBMM_EXP_CUBE_COMPUTE_TEMPLATE_PARAM
class WqbmmExpCubeCompute {
public:
    __aicore__ inline WqbmmExpCubeCompute(){};
    __aicore__ inline void LaunchMatmul(uint64_t nOffset, uint64_t kGmOffset, const A16W4MsdConstParam &constParams,
                                        const GlobalTensor<int8_t> &aUnfoldGm, const GlobalTensor<int32_t> &yS32Global);
    __aicore__ inline void Init(GM_ADDR weight, const TCubeTiling *matmulTiling, TPipe *tPipe);

private:
    static constexpr uint64_t L1_BUF_NUM = 2;
    static constexpr uint64_t L1_BUFFER_OFFSET = 128 * 1024;

    GlobalTensor<int8_t> weightS8Gm_;  // int8伪装2个int4

    LocalTensor<int8_t> aUnfoldS8L1_;
    LocalTensor<int8_t> wS8L1_;
    uint64_t l1LoopIdx_ = 0;

    using InputXType = MatmulType<TPosition::A1, CubeFormat::NZ, wType, false>;
    using InputWType = MatmulType<TPosition::B1, CubeFormat::NZ, wType, false>;
    using OutputYType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using InputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    MatmulImpl<InputXType, InputWType, OutputYType, InputBiasType> mmObj_;

    __aicore__ inline void CopyGmToL1Nd2Nz(uint64_t nOffset, uint64_t kGmOffset, const A16W4MsdConstParam &constParams,
                                           const GlobalTensor<int8_t> &aUnfoldGm);
};

WQBMM_EXP_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_CUBE_COMPUTE_CLASS::Init(GM_ADDR weight, const TCubeTiling *matmulTiling, TPipe *tPipe)
{
    weightS8Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight));

    TBuf<TPosition::A1> l1TBuf;
    tPipe->InitBuffer(l1TBuf, 512 * 1024); // l1总共512 KB
    aUnfoldS8L1_ = l1TBuf.Get<int8_t>();
    wS8L1_ = l1TBuf.Get<int8_t>()[256 * 1024]; // weight从256k开始
    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(matmulTiling, tPipe);
}

WQBMM_EXP_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_CUBE_COMPUTE_CLASS::LaunchMatmul(uint64_t nOffset, uint64_t kGmOffset,
                                                                  const A16W4MsdConstParam &constParams,
                                                                  const GlobalTensor<int8_t> &aUnfoldGm,
                                                                  const GlobalTensor<int32_t> &yS32Global)
{
    CopyGmToL1Nd2Nz(nOffset, kGmOffset, constParams, aUnfoldGm);
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);

    mmObj_.SetOrgShape(UNFLOD_TIMES * constParams.mGmSize, constParams.nGmSize, constParams.kaL1Size,
                       constParams.kbL1Size, constParams.nGmSize);
    mmObj_.SetTensorA(aUnfoldS8L1_[(l1LoopIdx_ % L1_BUF_NUM) * L1_BUFFER_OFFSET].template ReinterpretCast<int4b_t>(),
                      false);
    mmObj_.SetTensorB(wS8L1_[(l1LoopIdx_ % L1_BUF_NUM) * L1_BUFFER_OFFSET].template ReinterpretCast<int4b_t>(), false);
    mmObj_.SetTail(constParams.mL1Size, constParams.nL1Size, constParams.kbL1Size);
    mmObj_.IterateAll(yS32Global);
    mmObj_.End();
    l1LoopIdx_++;
}

WQBMM_EXP_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_CUBE_COMPUTE_CLASS::CopyGmToL1Nd2Nz(uint64_t nOffset, uint64_t kGmOffset,
                                                                     const A16W4MsdConstParam &constParams,
                                                                     const GlobalTensor<int8_t> &aUnfoldGm)
{
    // B矩阵 k,n
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.nValue = constParams.kbL1Size;
    nd2nzPara.dValue = constParams.nL1Size >> 1;
    nd2nzPara.srcDValue = constParams.nGmSize >> 1;
    nd2nzPara.dstNzC0Stride = CeilAlign(nd2nzPara.nValue, static_cast<uint16_t>(BLOCK_CUBE));
    DataCopy(wS8L1_[(l1LoopIdx_ % L1_BUF_NUM) * L1_BUFFER_OFFSET],
             weightS8Gm_[(nOffset >> 1) + kGmOffset * (constParams.nGmSize >> 1)], nd2nzPara);

    // A矩阵 m,k
    nd2nzPara.nValue = constParams.mL1Size;
    nd2nzPara.dValue = constParams.kbL1Size >> 1;
    nd2nzPara.srcDValue = constParams.kbL1Size >> 1;
    nd2nzPara.dstNzC0Stride = CeilAlign(nd2nzPara.nValue, static_cast<uint16_t>(BLOCK_CUBE));  // 对齐到16 单位block
    // 默认一块buf最多放两份
    DataCopy(aUnfoldS8L1_[(l1LoopIdx_ % L1_BUF_NUM) * L1_BUFFER_OFFSET], aUnfoldGm, nd2nzPara);
}
}  // namespace WeightQuantBatchMatmulExperimental
#endif