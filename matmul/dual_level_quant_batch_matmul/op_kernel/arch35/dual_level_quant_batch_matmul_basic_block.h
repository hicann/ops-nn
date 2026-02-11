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
 * \file dual_level_quant_batch_matmul_basic_block.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "dual_level_quant_batch_matmul_block.h"
#include "dual_level_quant_batch_matmul_cube_compute.h"
#include "dual_level_quant_batch_matmul_cube_compute_tools.h"
#include "dual_level_quant_batch_matmul_vec_compute.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::LocalTensor;
using AscendC::TPosition;

#ifdef LOCAL_TEMPLATE_CLASS_PARAMS
#undef LOCAL_TEMPLATE_CLASS_PARAMS
#endif
#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                             \
    template <                                                                                                  \
        typename x1Type, typename x2Type, typename x1Level1ScaleType, typename x1Level0ScaleType,               \
        typename x2Level1ScaleType, typename x2Level0ScaleType, typename biasType, typename yType, bool aTrans, \
        bool bTrans, bool hasBias>

#ifdef LOCAL_TEMPLATE_FUNC_PARAMS
#undef LOCAL_TEMPLATE_FUNC_PARAMS
#endif
#define LOCAL_TEMPLATE_FUNC_PARAMS                                                                               \
    x1Type, x2Type, x1Level1ScaleType, x1Level0ScaleType, x2Level1ScaleType, x2Level0ScaleType, biasType, yType, \
        aTrans, bTrans, hasBias

namespace DualLevelQuantBatchMatmul::Arch35 {

enum SyncMode
{
    SYNC_MODE0 = 0, // C/V全核同步
    SYNC_MODE1 = 1, // AIV之间的同步
    SYNC_MODE2 = 2, // CV之间的同步
    SYNC_MODE4 = 4, // CV分离的同步
};

LOCAL_TEMPLATE_CLASS_PARAMS
class DualLevelQuantMatmulBasicBlock {
public:
    __aicore__ inline DualLevelQuantMatmulBasicBlock() = default;

    __aicore__ inline void Init(
        __gm__ x1Type* x1, __gm__ x2Type* x2, __gm__ x1Level0ScaleType* x1Level0Scale,
        __gm__ x1Level1ScaleType* x1Level1Scale, __gm__ x2Level0ScaleType* x2Level0Scale,
        __gm__ x2Level1ScaleType* x2Level1Scale, __gm__ biasType* bias, __gm__ yType* y,
        const DualLevelQuantBatchMatmulBasicTilingData* tilingData);

    __aicore__ inline void SetAivToAic();

    __aicore__ inline void WaitAivToAic();

    __aicore__ inline void ComputeBasicBlock(
        DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params);

    __aicore__ inline void ComputeAicBasicBlock(
        DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params, FixL0CToDstParams& fixpParams,
        uint64_t kGmOffset);

    __aicore__ inline void ComputeAivBasicBlock(
        DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params, uint64_t kGmOffset);

    __aicore__ inline void End();

protected:
    static constexpr uint64_t AIV_SYNC_FLAG_DIST = 16;
    static constexpr uint64_t AIV_TO_AIC_ID = 2;
    static constexpr uint64_t AIC_TO_AIV_ID = 3;

    static constexpr uint64_t INT4_ONE_BLK_SIZE = 64;
    static constexpr uint64_t CV_LOOP_NUM = 2;
    static constexpr uint64_t C_TMP_FP32_UB_OFFSET = 64 * 128;
    LocalTensor<float> cTmpFp32Ub_{TPosition::VECIN, 0, 248 * 1024};

    DualLevelQuantBatchMatmulVectorCompute<x1Level0ScaleType, x2Level0ScaleType, biasType, yType, hasBias> vecCompute_;
    DualLevelQuantBatchMatmulCubeCompute<x1Type, x2Type, x1Level1ScaleType, x2Level1ScaleType, float> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    __gm__ x1Type* x1, __gm__ x2Type* x2, __gm__ x1Level0ScaleType* x1Level0Scale,
    __gm__ x1Level1ScaleType* x1Level1Scale, __gm__ x2Level0ScaleType* x2Level0Scale,
    __gm__ x2Level1ScaleType* x2Level1Scale, __gm__ biasType* bias, __gm__ yType* y,
    const DualLevelQuantBatchMatmulBasicTilingData* tilingData)
{
    if ASCEND_IS_AIC {
        cubeCompute_.Init(x1, x2, x1Level1Scale, x2Level1Scale);
    } else {
        vecCompute_.Init(x1Level0Scale, x2Level0Scale, bias, y);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicBlock(
    DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params)
{
    FixL0CToDstParams fixpParams;
    fixpParams.mL0Size = l0Params.mL0Size;
    fixpParams.nL0Size = l0Params.nL0Size;
    fixpParams.outNSize = fixpParams.nL0Size;

    offsetParams.mUbSize = Ops::Base::CeilDiv<uint64_t>(l0Params.mL0Size, 2);

    for (uint64_t kGmOffset = 0; kGmOffset < offsetParams.kSize; kGmOffset += offsetParams.l0BaseK) {
        offsetParams.kGmOffset = kGmOffset;

        uint64_t kGmRealSize = kGmOffset + offsetParams.l0BaseK > offsetParams.kSize ? offsetParams.kSize - kGmOffset :
                                                                                       offsetParams.l0BaseK;
        l0Params.kL1Size = kGmRealSize;
        l0Params.kL0Size = kGmRealSize;

        if ASCEND_IS_AIC {
            ComputeAicBasicBlock(offsetParams, l0Params, fixpParams, kGmOffset);
        } else {
            ComputeAivBasicBlock(offsetParams, l0Params, kGmOffset);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeAicBasicBlock(
    DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params, FixL0CToDstParams& fixpParams,
    uint64_t kGmOffset)
{
    cubeCompute_.WaitMte1ToMte2(offsetParams);
    cubeCompute_.CopyGmToL1(offsetParams, l0Params);
    for (uint64_t nL1Offset = 0; nL1Offset < l0Params.nL1Size; nL1Offset += l0Params.nL0Size) {
        for (uint64_t mL1Offset = 0; mL1Offset < l0Params.mL1Size; mL1Offset += l0Params.mL0Size) {
            CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID);
            CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID + AIV_SYNC_FLAG_DIST);

            cubeCompute_.LaunchMatmul(
                mL1Offset, nL1Offset, 0, kGmOffset % offsetParams.level1ScaleKL1Size,
                cTmpFp32Ub_[cvLoopIdx_ % CV_LOOP_NUM * C_TMP_FP32_UB_OFFSET], l0Params, fixpParams);

            CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIC_TO_AIV_ID);
            CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIC_TO_AIV_ID + AIV_SYNC_FLAG_DIST);
            cvLoopIdx_++;
        }
    }
    cubeCompute_.SetMte1ToMte2(offsetParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeAivBasicBlock(
    DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params, uint64_t kGmOffset)
{
    if (kGmOffset % offsetParams.level0ScaleKUbSize == 0) {
        vecCompute_.WaitVToMte2ForX1();
        vecCompute_.CopyX1Level0ScaleToUb(offsetParams, l0Params);
    }
    vecCompute_.WaitVToMte2ForX2();
    vecCompute_.CopyX2Level0ScaleAndBiasToUb(offsetParams, l0Params);
    vecCompute_.SetAndWaitMte2ToV();
    for (uint64_t nL1Offset = 0; nL1Offset < l0Params.nL1Size; nL1Offset += l0Params.nL0Size) {
        uint64_t cFp32BufId = nL1Offset / l0Params.nL0Size;
        if (kGmOffset == 0) {
            vecCompute_.WaitMte3ToV(cFp32BufId);
            vecCompute_.InitUb(cFp32BufId, nL1Offset);
        }
        for (uint64_t mL1Offset = 0; mL1Offset < l0Params.mL1Size; mL1Offset += l0Params.mL0Size) {
            CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIC_TO_AIV_ID);
            vecCompute_.AntiQuantCompute(
                cFp32BufId, cvLoopIdx_, mL1Offset, nL1Offset, cTmpFp32Ub_, offsetParams, l0Params);
            CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIV_TO_AIC_ID);
            cvLoopIdx_++;
            if (kGmOffset + offsetParams.l0BaseK >= offsetParams.kSize) {
                vecCompute_.CopyUbToGm(cFp32BufId, mL1Offset, nL1Offset, offsetParams, l0Params);
            }
        }
        if (kGmOffset + offsetParams.l0BaseK >= offsetParams.kSize) {
            vecCompute_.SetMte3ToV(cFp32BufId);
        }
    }

    if ((kGmOffset + offsetParams.l0BaseK) % offsetParams.level0ScaleKUbSize == 0 ||
        kGmOffset + offsetParams.l0BaseK >= offsetParams.kSize) {
        vecCompute_.SetVToMte2ForX1();
    }
    vecCompute_.SetVToMTE2ForX2();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::SetAivToAic()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIV_TO_AIC_ID);
        CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIV_TO_AIC_ID);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::WaitAivToAic()
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID);
        CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID + AIV_SYNC_FLAG_DIST);
        CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID);
        CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID + AIV_SYNC_FLAG_DIST);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        cubeCompute_.EndSync();
    } else {
        vecCompute_.WaitVectorFlag();
    }
}

} // namespace DualLevelQuantBatchMatmul::Arch35
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_H
