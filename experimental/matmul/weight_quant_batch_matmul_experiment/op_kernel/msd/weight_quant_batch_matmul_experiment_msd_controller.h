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
 * \file weight_quant_batch_matmul_experimental_msd_controller.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_MSD_CONTROLLER_H
#define WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_MSD_CONTROLLER_H

#include "../weight_quant_batch_matmul_experiment_tiling_data.h"
#include "../weight_quant_batch_matmul_experiment_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_experiment_cube_compute.h"
#include "weight_quant_batch_matmul_experiment_tool.h"
#include "weight_quant_batch_matmul_experiment_vec_compute.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::TPipe;

namespace WeightQuantBatchMatmulExperimental {

#define WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM template <typename xType, typename wType, int msdMode>

#define WQBMM_EXP_MSD_CONTROLLER_CLASS WqbmmExpMsdController<xType, wType, msdMode>

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
class WqbmmExpMsdController {
public:
    __aicore__ inline WqbmmExpMsdController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR y, GM_ADDR workspace,
                                const WeightQuantBatchMatmulExperimentTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    const WeightQuantBatchMatmulExperimentTilingData *tilingData_;
    WqbmmExpCubeCompute<xType, wType, int32_t> cubeCompute_;
    WqbmmExpVecCompute<xType, wType, int32_t> vecCompute_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<float> aMaxWorkspaceGlobal_;
    GlobalTensor<int32_t> yS32Gm_;
    GlobalTensor<wType> aUnfoldS4Global_;
    GlobalTensor<int8_t> aUnfoldS8Global_;
    __aicore__ inline void BasicMsd(uint64_t nGmOffset, uint64_t kGmOffset, const A16W4MsdConstParam &constParams);
    __aicore__ inline void BasicEnd();

    __aicore__ inline void PreloadMsd(uint64_t nGmOffset, uint64_t curKGmOffset, uint64_t lastKGmOffset,
                                      const A16W4MsdConstParam &constParams);
    __aicore__ inline void PreloadEnd(uint64_t nGmOffset, uint64_t lastKGmOffset,
                                      const A16W4MsdConstParam &constParams);
    uint64_t cvLoopIdx_ = 0;
    uint64_t aMaxWsOffset_;
    uint64_t aUnfoldWsS4Offset_;
    uint64_t aUnfoldWsS8Offset_;
    uint64_t yS32WsOffset_;
};

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_MSD_CONTROLLER_CLASS::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR y, GM_ADDR workspace,
    const WeightQuantBatchMatmulExperimentTilingData *tilingData, TPipe *tPipe)
{
    uint64_t bufferNum = 1;
    if constexpr (msdMode == PRELOAD_MSD) {
        bufferNum = DOUBLE_BUFFER_NUM;
    }

    // 设置workspace地址
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x));

    uint64_t wsB8Offset = 0;
    aMaxWorkspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    aMaxWsOffset_ = CeilAlign(tilingData->matmulTiling.M * FP32_BLOCK_SIZE,
                              512UL / sizeof(float));  // 空间分配按照512B对齐
    wsB8Offset += bufferNum * aMaxWsOffset_ * sizeof(float);

    // 实际表示同一个gm地址
    aUnfoldS4Global_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(workspace + wsB8Offset));
    aUnfoldS8Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace + wsB8Offset));

    aUnfoldWsS8Offset_ = CeilAlign(tilingData->matmulTiling.M * UNFLOD_TIMES * (tilingData->matmulTiling.Ka >> 1),
                                   512UL);  // 空间分配按照512B对齐;
    aUnfoldWsS4Offset_ = aUnfoldWsS8Offset_ << 1;
    wsB8Offset += 3 * aUnfoldWsS8Offset_;

    yS32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + wsB8Offset));
    yS32WsOffset_ = tilingData->matmulTiling.singleCoreM * tilingData->matmulTiling.N;

    // 初始化计算单元
    if ASCEND_IS_AIV {
        vecCompute_.Init(antiquantScale, y, tPipe);
    } else {
        cubeCompute_.Init(weight, &(tilingData->matmulTiling), tPipe);
    }

    // 设置反向同步依赖
    if ASCEND_IS_AIV {
        for (uint64_t i = 0; i < bufferNum; i++) {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIV_ONLY_AMAX_FLAG);
        }
    } else {
        if constexpr (msdMode == PRELOAD_MSD) {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIC_ONLY_AUNFOLD_FLAG);
        }
    }

    // 设置tiling data
    tilingData_ = tilingData;
}

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_MSD_CONTROLLER_CLASS::Process()
{
    uint64_t cubeBlockIdx = 0;
    cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx / 2;
    }
    // 当前模板简化实现，认为m方向不分核，n方向多核均分
    uint64_t nGmOffset = tilingData_->matmulTiling.baseN * cubeBlockIdx;

    A16W4MsdConstParam constParams;
    constParams.kaL1Size = tilingData_->matmulTiling.baseK;
    constParams.kbL1Size = tilingData_->matmulTiling.baseK;
    constParams.kUbSize = tilingData_->matmulTiling.baseK;

    constParams.kGmSize = tilingData_->matmulTiling.Ka;
    constParams.mGmSize = tilingData_->matmulTiling.M;
    constParams.nGmSize = tilingData_->matmulTiling.N;
    constParams.mL1Size = tilingData_->matmulTiling.singleCoreM;
    constParams.nL1Size = tilingData_->matmulTiling.singleCoreN;
    constParams.groupSize = 128;
    uint64_t kGmOffset = 0;
    for (; kGmOffset < tilingData_->matmulTiling.Ka; kGmOffset += tilingData_->matmulTiling.baseK, cvLoopIdx_++) {
        if constexpr (msdMode == PRELOAD_MSD) {
            PreloadMsd(nGmOffset, kGmOffset, cvLoopIdx_ > 0 ? kGmOffset - tilingData_->matmulTiling.baseK : 0,
                       constParams);
        } else {
            BasicMsd(nGmOffset, kGmOffset, constParams);
        }
    }

    if constexpr (msdMode == PRELOAD_MSD) {
        PreloadEnd(nGmOffset, kGmOffset - tilingData_->matmulTiling.baseK, constParams);
    } else {
        BasicEnd();
    }
}

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_MSD_CONTROLLER_CLASS::BasicMsd(uint64_t nGmOffset, uint64_t kGmOffset,
                                                                const A16W4MsdConstParam &constParams)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(SYNC_AIV_ONLY_AMAX_FLAG);
        if (GetBlockIdx() < tilingData_->matmulTiling.M) {
            vecCompute_.UnfoldA(1, GetBlockIdx(), constParams,
                                xGlobal_[GetBlockIdx() * tilingData_->matmulTiling.Ka + kGmOffset],
                                aMaxWorkspaceGlobal_[GetBlockIdx() * FP32_BLOCK_SIZE],
                                aUnfoldS4Global_[GetBlockIdx() * constParams.groupSize]);
        }
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_ONLY_A_UNFOLD_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_A_UNFOLD_FLAG);

        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);

        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
        vecCompute_.MergeY(0, nGmOffset, kGmOffset,
                           kGmOffset + tilingData_->matmulTiling.baseK >= tilingData_->matmulTiling.Ka,
                           aMaxWorkspaceGlobal_, yS32Gm_[nGmOffset], constParams);
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIV_ONLY_AMAX_FLAG);
    } else {
        CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
        cubeCompute_.LaunchMatmul(nGmOffset, kGmOffset, constParams, aUnfoldS8Global_, yS32Gm_[nGmOffset]);
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIC_ONLY_AUNFOLD_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_ONLY_AUNFOLD_FLAG);

        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }
}

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_MSD_CONTROLLER_CLASS::PreloadMsd(uint64_t nGmOffset, uint64_t curKGmOffset,
                                                                  uint64_t lastKGmOffset,
                                                                  const A16W4MsdConstParam &constParams)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(SYNC_AIV_ONLY_AMAX_FLAG);  // 等待AMax的写入flag
        if (GetBlockIdx() < tilingData_->matmulTiling.M) {
            vecCompute_.UnfoldA(
                1, GetBlockIdx(), constParams, xGlobal_[GetBlockIdx() * tilingData_->matmulTiling.Ka + curKGmOffset],
                aMaxWorkspaceGlobal_[cvLoopIdx_ % DOUBLE_BUFFER_NUM * aMaxWsOffset_ + GetBlockIdx() * FP32_BLOCK_SIZE],
                aUnfoldS4Global_[cvLoopIdx_ % 3 * aUnfoldWsS4Offset_ + GetBlockIdx() * constParams.groupSize]);
        }
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_ONLY_A_UNFOLD_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_A_UNFOLD_FLAG);

        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);

        if (cvLoopIdx_ > 0) {
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            vecCompute_.MergeY(0, nGmOffset, lastKGmOffset,
                               lastKGmOffset + tilingData_->matmulTiling.baseK >= tilingData_->matmulTiling.Ka,
                               aMaxWorkspaceGlobal_[(cvLoopIdx_ - 1) % DOUBLE_BUFFER_NUM * aMaxWsOffset_],
                               yS32Gm_[(cvLoopIdx_ - 1) % DOUBLE_BUFFER_NUM * yS32WsOffset_ + nGmOffset], constParams);
            CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIV_ONLY_AMAX_FLAG);
        }
    } else {
        CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);

        cubeCompute_.LaunchMatmul(nGmOffset, curKGmOffset, constParams,
                                  aUnfoldS8Global_[cvLoopIdx_ % 3 * aUnfoldWsS8Offset_],
                                  yS32Gm_[cvLoopIdx_ % DOUBLE_BUFFER_NUM * yS32WsOffset_ + nGmOffset]);

        CrossCoreWaitFlag(SYNC_AIC_ONLY_AUNFOLD_FLAG);  // 等上一轮的AIC消费完

        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);

        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIC_ONLY_AUNFOLD_FLAG);
    }
}

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_MSD_CONTROLLER_CLASS::PreloadEnd(uint64_t nGmOffset, uint64_t lastKGmOffset,
                                                                  const A16W4MsdConstParam &constParams)
{
    if ASCEND_IS_AIV {
        if (cvLoopIdx_ > 0) {
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            vecCompute_.MergeY(0, nGmOffset, lastKGmOffset,
                               lastKGmOffset + tilingData_->matmulTiling.baseK >= tilingData_->matmulTiling.Ka,
                               aMaxWorkspaceGlobal_[(cvLoopIdx_ - 1) % DOUBLE_BUFFER_NUM * aMaxWsOffset_],
                               yS32Gm_[(cvLoopIdx_ - 1) % DOUBLE_BUFFER_NUM * yS32WsOffset_ + nGmOffset], constParams);
            CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIV_ONLY_AMAX_FLAG);
        }

        CrossCoreWaitFlag(SYNC_AIV_ONLY_AMAX_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_AMAX_FLAG);
        vecCompute_.End();
    } else {
        CrossCoreWaitFlag(SYNC_AIC_ONLY_AUNFOLD_FLAG);
    }
}

WQBMM_EXP_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_MSD_CONTROLLER_CLASS::BasicEnd()
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(SYNC_AIV_ONLY_AMAX_FLAG);
        vecCompute_.End();
    }
}
}  // namespace WeightQuantBatchMatmulExperimental
#endif  // weight_quant_batch_matmul_experimental_msd_controller