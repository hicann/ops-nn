/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_BATCH_NORM_GRAD_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_BATCH_NORM_GRAD_H_
#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t INDEX_0 = 0;
constexpr int64_t INDEX_1 = 1;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;
constexpr int64_t INDEX_4 = 4;
constexpr int64_t INDEX_5 = 5;

static constexpr int64_t DIM_NUM_4 = 4;
static constexpr int64_t DIM_NUM_5 = 5;

static constexpr int64_t INPUT_NUM = 6;

// Attr属性 index
constexpr int64_t PARAM_ATTRS_EPSILON_INDEX = 0;
constexpr int64_t PARAM_ATTRS_DATA_FORMAT_INDEX = 1;
constexpr int64_t PARAM_ATTRS_ISTRAING_INDEX = 2;
constexpr int64_t PARAM_ATTRS_OUTPUT_MASK = 3;

constexpr int64_t PARAM_INPUT_DY = 0;                       // dy
constexpr int64_t PARAM_INPUT_WEIGHT_INDEX = 2;             // wweight、scale
constexpr int64_t PARAM_INPUT_RUNNINGVAR_INDEX = 4;         // reserve_space2  train模式：rstd，infer模式：var
constexpr int64_t PARAM_OUTPUT_DX_INDEX = 0;                // x_backprop

constexpr float DEFAULT_EPSILON = 1e-4;

BEGIN_TILING_DATA_DEF(BatchNormGradBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, r1Dim);          // R1AR0 R外轴
TILING_DATA_FIELD_DEF(int64_t, aDim);           // R1AR0 A轴
TILING_DATA_FIELD_DEF(int64_t, r0Dim);          // R1AR0 R0内轴
TILING_DATA_FIELD_DEF(int64_t, rAlign);         // 用于随路转置为AR，R轴对齐
TILING_DATA_FIELD_DEF(int64_t, blockNum);       // 参与计算的所有核数
TILING_DATA_FIELD_DEF(int64_t, tailBlockNum);   // 尾核的个数
TILING_DATA_FIELD_DEF(int64_t, formerBlockDim); // 整核分配的A轴个数，aDim/blockNum 向下取整
TILING_DATA_FIELD_DEF(int64_t, tailBlockDim);   // 尾核分配的A轴个数，aDim/blockNum 向上取整
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGradBaseTilingDataOp, BatchNormGradBaseTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, dummy);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad, BatchNormGradTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradBinaryAddTilingData)
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddk);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGradBinaryAddTilingDataOp, BatchNormGradBinaryAddTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRARFullLoadTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBaseTilingData, baseTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, binaryAddTilingData);
TILING_DATA_FIELD_DEF(int64_t, formerUbDim);         // 一次完整的UB内，循环执行A轴的次数
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock); // 整核进行UB循环的次数，formerBlockDim/formerUbDim 向上取整
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock); // 整核最后一轮UB循环，处理的A轴个数
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock); // 尾核进行UB循环的次数，tailBlockDim/formerUbDim 向上取整
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock); // 尾核最后一轮UB循环，处理的A轴个数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_10000000, BatchNormGradRARFullLoadTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRARRecomputeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBaseTilingData, baseTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, generalBinAddTilingData); // 整块的二分累加参数
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, tailBinAddTilingData); // 整块+尾块的二分累加参数
TILING_DATA_FIELD_DEF(int64_t, ubRDimFactor);              // 一次完整的UB内，循环执行R轴的个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimFactorAlign);         // 一次完整的UB内，循环对齐到block的R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubRDimLoopNum);             // 核内R轴UB循环的次数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTail);                // R轴尾块大小，R - ubRDimFactor * ubRDimLoopNum
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailFactor);          // R轴尾块一次循环处理的R轴个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailFactorAlign);     // R轴尾块一次循环对齐到block的R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailLoopNum);         // R轴尾块循环次数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTail);            // R轴尾块的尾块的个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTailFactor);      // R轴尾块的尾块一次循环处理的R轴个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTailFactorAlign); // R轴尾块的尾块一次循环对齐到block的R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTailLoopNum);     // R轴尾块的尾块的循环次数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_31000000, BatchNormGradRARRecomputeTilingData);
REGISTER_TILING_DATA_CLASS(BatchNormGrad_32000000, BatchNormGradRARRecomputeTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRAFullLoadTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBaseTilingData, baseTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, binaryAddTilingData);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, mainBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, mainBlockCount);
TILING_DATA_FIELD_DEF(int64_t, tailBlockCount);
TILING_DATA_FIELD_DEF(int64_t, mainALoopFactor);
TILING_DATA_FIELD_DEF(int64_t, mainALoopFactorAligned);
TILING_DATA_FIELD_DEF(int64_t, tailALoopFactor);
TILING_DATA_FIELD_DEF(int64_t, tailALoopFactorAligned);
TILING_DATA_FIELD_DEF(int64_t, foldLoopStep1);
TILING_DATA_FIELD_DEF(int64_t, foldLoopOffset1);
TILING_DATA_FIELD_DEF(int64_t, foldLoopStep2);
TILING_DATA_FIELD_DEF(int64_t, foldLoopOffset2);
TILING_DATA_FIELD_DEF(int64_t, foldLoopStep3);
TILING_DATA_FIELD_DEF(int64_t, foldLoopOffset3);
TILING_DATA_FIELD_DEF(int64_t, reduceRecursionLoop);
TILING_DATA_FIELD_DEF(int64_t, reduceLoopTimes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_20000000, BatchNormGradRAFullLoadTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRARecomputeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBaseTilingData, baseTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, binaryAddTilingData);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, mainBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, mainBlockCount);
TILING_DATA_FIELD_DEF(int64_t, tailBlockCount);
TILING_DATA_FIELD_DEF(int64_t, aLoopFactor);
TILING_DATA_FIELD_DEF(int64_t, aLoopFactorAligned);
TILING_DATA_FIELD_DEF(int64_t, rLoopFactor);
TILING_DATA_FIELD_DEF(int64_t, rLoopTimes);
TILING_DATA_FIELD_DEF(int64_t, rLoopTail);
TILING_DATA_FIELD_DEF(int64_t, binaryFoldPoint);
TILING_DATA_FIELD_DEF(int64_t, binaryBlockCount);
TILING_DATA_FIELD_DEF(int64_t, binaryTailBlock);
TILING_DATA_FIELD_DEF(int64_t, cacheBufferCount);
TILING_DATA_FIELD_DEF(float, reciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_40000000, BatchNormGradRARecomputeTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRASplitRTilingData)
TILING_DATA_FIELD_DEF(int64_t, rDim);
TILING_DATA_FIELD_DEF(int64_t, aDim);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);       // 需要多少个核
TILING_DATA_FIELD_DEF(int64_t, rLoopFactor);       // 一次循环处理多少个R
TILING_DATA_FIELD_DEF(int64_t, blockFactor);       // 每个核处理多少个R
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);   // 尾核处理多少个R
TILING_DATA_FIELD_DEF(int64_t, binaryBlockCnt);    // ub内rLoopFactor需要循环几次
TILING_DATA_FIELD_DEF(int64_t, binaryFoldPoint);   // ub内二分折叠点
TILING_DATA_FIELD_DEF(int64_t, binaryBlockTail);   // 最后一次处理多少个R
TILING_DATA_FIELD_DEF(int64_t, lastCoreBlockCnt);  // 尾核ub内rLoopFactor需要循环几次
TILING_DATA_FIELD_DEF(int64_t, lastCoreFoldPoint); // 尾核ub内二分折叠点
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoopTail);  // 尾核最后一次处理多少个R
TILING_DATA_FIELD_DEF(int64_t, aFactor);           // 一次循环可以A轴处理多少
TILING_DATA_FIELD_DEF(int64_t, aFactorAlign);      // a方向对齐大小
TILING_DATA_FIELD_DEF(int64_t, aFactorTail);       // 最后一次处理多少
TILING_DATA_FIELD_DEF(int64_t, aLoopTimes);        // A轴需要处理多少次
TILING_DATA_FIELD_DEF(int64_t, dxLoopFactor);      // 计算dx时一次处理多少行
TILING_DATA_FIELD_DEF(int64_t, dxLoopTail);        // 计算dx时尾块处理多少行
TILING_DATA_FIELD_DEF(int64_t, dxLoopTimes);       // 计算dx时R轴需要循环多少次
TILING_DATA_FIELD_DEF(int64_t, dxLastCoreFactor);  // 计算dx时尾核一次处理多少行
TILING_DATA_FIELD_DEF(int64_t, dxLastCoreTail);    // 计算dx时尾核尾块处理多少行
TILING_DATA_FIELD_DEF(int64_t, dxLastCoreTimes);   // 计算dx时尾核R轴需要循环多少次
TILING_DATA_FIELD_DEF(int64_t, cacheBuffCnt);      // cacheBuffer个数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_50000000, BatchNormGradRASplitRTilingData);

// inference
BEGIN_TILING_DATA_DEF(BatchNormGradInferChannelLastDxTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, aDim);
TILING_DATA_FIELD_DEF(int64_t, aOuter);
TILING_DATA_FIELD_DEF(int64_t, bOuter);
TILING_DATA_FIELD_DEF(int64_t, tileBlockALen);
TILING_DATA_FIELD_DEF(int64_t, tileBlockATail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockAPaddingNum);
TILING_DATA_FIELD_DEF(int64_t, tileBlockBLen);
TILING_DATA_FIELD_DEF(int64_t, tileBlockBTail);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGradInferChannelLastDxTilingDataOp, BatchNormGradInferChannelLastDxTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradInferChannelLastTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradInferChannelLastDxTilingData, dxTilingData);
TILING_DATA_FIELD_DEF(int64_t, binAddRFactorStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddRLoopStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddRTotalLoopStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddRTailStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddBasicBlockLoopStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddMainFoldCountStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddCacheBufferCountStg1);
TILING_DATA_FIELD_DEF(int64_t, binAddResultCacheIDStg1);
TILING_DATA_FIELD_DEF(int64_t, aDimStg1);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg1);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg1);
TILING_DATA_FIELD_DEF(int64_t, aTailStg1);
TILING_DATA_FIELD_DEF(int64_t, aOuterPerCoreStg1);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumsStg1);
TILING_DATA_FIELD_DEF(int32_t, enableDx);
TILING_DATA_FIELD_DEF(int32_t, enableDgamma);
TILING_DATA_FIELD_DEF(int32_t, enableDbeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_900000, BatchNormGradInferChannelLastTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradInferDxTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, r1Dim);
TILING_DATA_FIELD_DEF(int64_t, aDim);
TILING_DATA_FIELD_DEF(int64_t, r0Dim);
TILING_DATA_FIELD_DEF(int64_t, r1Outer);
TILING_DATA_FIELD_DEF(int64_t, aOuter);
TILING_DATA_FIELD_DEF(int64_t, r0Outer);
TILING_DATA_FIELD_DEF(int64_t, tileBlockR1Len);
TILING_DATA_FIELD_DEF(int64_t, tileBlockR1Tail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockALen);
TILING_DATA_FIELD_DEF(int64_t, tileBlockATail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockR0Len);
TILING_DATA_FIELD_DEF(int64_t, tileBlockR0Tail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockAPaddingNum);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGradInferDxTilingDataOp, BatchNormGradInferDxTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradInferTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradInferDxTilingData, baseTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, generalBinAddTilingData); // 整块的二分累加参数
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBinaryAddTilingData, tailBinAddTilingData); // 整块+尾块的二分累加参数
TILING_DATA_FIELD_DEF(int64_t, blockNum);       // 参与计算的所有核数
TILING_DATA_FIELD_DEF(int64_t, tailBlockNum);   // 尾核的个数
TILING_DATA_FIELD_DEF(int64_t, formerBlockDim); // 整核分配的A轴个数，aDim/blockNum 向下取整
TILING_DATA_FIELD_DEF(int64_t, tailBlockDim);   // 尾核分配的A轴个数，aDim/blockNum 向上取整
TILING_DATA_FIELD_DEF(int64_t, ubRDimFactor);              // 一次完整的UB内，循环执行R轴的个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimFactorAlign);         // 一次完整的UB内，循环对齐到block的R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubRDimLoopNum);             // 核内R轴UB循环的次数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTail);                // R轴尾块大小，R - ubRDimFactor * ubRDimLoopNum
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailFactor);          // R轴尾块一次循环处理的R轴个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailFactorAlign);     // R轴尾块一次循环对齐到block的R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailLoopNum);         // R轴尾块循环次数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTail);            // R轴尾块的尾块的个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTailFactor);      // R轴尾块的尾块一次循环处理的R轴个数
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTailFactorAlign); // R轴尾块的尾块一次循环对齐到block的R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubRDimTailTailLoopNum);     // R轴尾块的尾块的循环次数
TILING_DATA_FIELD_DEF(int32_t, enableDx);
TILING_DATA_FIELD_DEF(int32_t, enableDgamma);
TILING_DATA_FIELD_DEF(int32_t, enableDbeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_910001, BatchNormGradInferTilingData); // SplitR1
REGISTER_TILING_DATA_CLASS(BatchNormGrad_910002, BatchNormGradInferTilingData); // SplitR0

BEGIN_TILING_DATA_DEF(BatchNormGradFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, b1Dim);
TILING_DATA_FIELD_DEF(int64_t, aDim);
TILING_DATA_FIELD_DEF(int64_t, b0Dim);
TILING_DATA_FIELD_DEF(int64_t, bAlign);
TILING_DATA_FIELD_DEF(int64_t, coreChannelNum);
TILING_DATA_FIELD_DEF(int64_t, coreChannelNumTail);
TILING_DATA_FIELD_DEF(int64_t, cUbBlock);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_10000, BatchNormGradFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(BatchNormGrad_30000, BatchNormGradFullLoadTilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRARSplitCoreR1TilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBaseTilingData, baseTilingData);
TILING_DATA_FIELD_DEF(int64_t, r1Dim);
TILING_DATA_FIELD_DEF(int64_t, aDim);
TILING_DATA_FIELD_DEF(int64_t, aDimAligned);
TILING_DATA_FIELD_DEF(int64_t, r0Dim);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, r1Inner);
TILING_DATA_FIELD_DEF(int64_t, r1Tail);
TILING_DATA_FIELD_DEF(int64_t, r0InnerStg0);
TILING_DATA_FIELD_DEF(int64_t, r0OuterStg0);
TILING_DATA_FIELD_DEF(int64_t, r0TailStg0);
TILING_DATA_FIELD_DEF(int64_t, r0TailAlignedStg0);
TILING_DATA_FIELD_DEF(int64_t, r1InnerInnerStg0);
TILING_DATA_FIELD_DEF(int64_t, r1InnerOuterStg0);
TILING_DATA_FIELD_DEF(int64_t, r1InnerTailStg0);
TILING_DATA_FIELD_DEF(int64_t, r1TailOuterStg0);
TILING_DATA_FIELD_DEF(int64_t, r1TailTailStg0);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg0);
TILING_DATA_FIELD_DEF(int64_t, aInnerAlignedStg0);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg0);
TILING_DATA_FIELD_DEF(int64_t, aTailStg0);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg1);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg1);
TILING_DATA_FIELD_DEF(int64_t, aTailStg1);
TILING_DATA_FIELD_DEF(int64_t, r0InnerStg2);
TILING_DATA_FIELD_DEF(int64_t, r0OuterStg2);
TILING_DATA_FIELD_DEF(int64_t, r0TailStg2);
TILING_DATA_FIELD_DEF(int64_t, r0TailAlignedStg2);
TILING_DATA_FIELD_DEF(int64_t, r1InnerInnerStg2);
TILING_DATA_FIELD_DEF(int64_t, r1InnerOuterStg2);
TILING_DATA_FIELD_DEF(int64_t, r1InnerTailStg2);
TILING_DATA_FIELD_DEF(int64_t, r1TailOuterStg2);
TILING_DATA_FIELD_DEF(int64_t, r1TailTailStg2);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg2);
TILING_DATA_FIELD_DEF(int64_t, aInnerAlignedStg2);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg2);
TILING_DATA_FIELD_DEF(int64_t, aTailStg2);
TILING_DATA_FIELD_DEF(int64_t, binAddBasicBlockLoop);
TILING_DATA_FIELD_DEF(int64_t, binAddMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, binAddCacheBufferCount);
TILING_DATA_FIELD_DEF(int64_t, binAddResultCacheID);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBinAddBasicBlockLoop);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBinAddMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBinAddResultCacheID);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_1000, BatchNormGradRARSplitCoreR1TilingData);

BEGIN_TILING_DATA_DEF(BatchNormGradRARSplitCoreR0TilingData)
TILING_DATA_FIELD_DEF_STRUCT(BatchNormGradBaseTilingData, baseTilingData);
TILING_DATA_FIELD_DEF(int64_t, r1Dim);
TILING_DATA_FIELD_DEF(int64_t, aDim);
TILING_DATA_FIELD_DEF(int64_t, aDimAligned);
TILING_DATA_FIELD_DEF(int64_t, r0Dim);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, r0Inner);
TILING_DATA_FIELD_DEF(int64_t, r0Tail);
TILING_DATA_FIELD_DEF(int64_t, r0InnerInnerStg0);
TILING_DATA_FIELD_DEF(int64_t, r0InnerOuterStg0);
TILING_DATA_FIELD_DEF(int64_t, r0InnerTailStg0);
TILING_DATA_FIELD_DEF(int64_t, r0TailOuterStg0);
TILING_DATA_FIELD_DEF(int64_t, r0TailTailStg0);
TILING_DATA_FIELD_DEF(int64_t, r0TailTailAlignedStg0);
TILING_DATA_FIELD_DEF(int64_t, r1InnerStg0);
TILING_DATA_FIELD_DEF(int64_t, r1OuterStg0);
TILING_DATA_FIELD_DEF(int64_t, r1TailStg0);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg0);
TILING_DATA_FIELD_DEF(int64_t, aInnerAlignedStg0);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg0);
TILING_DATA_FIELD_DEF(int64_t, aTailStg0);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg1);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg1);
TILING_DATA_FIELD_DEF(int64_t, aTailStg1);
TILING_DATA_FIELD_DEF(int64_t, r1InnerStg2);
TILING_DATA_FIELD_DEF(int64_t, r1OuterStg2);
TILING_DATA_FIELD_DEF(int64_t, r1TailStg2);
TILING_DATA_FIELD_DEF(int64_t, r0InnerInnerStg2);
TILING_DATA_FIELD_DEF(int64_t, r0InnerOuterStg2);
TILING_DATA_FIELD_DEF(int64_t, r0InnerTailStg2);
TILING_DATA_FIELD_DEF(int64_t, r0TailOuterStg2);
TILING_DATA_FIELD_DEF(int64_t, r0TailTailStg2);
TILING_DATA_FIELD_DEF(int64_t, r0TailTailAlignedStg2);
TILING_DATA_FIELD_DEF(int64_t, aInnerStg2);
TILING_DATA_FIELD_DEF(int64_t, aInnerAlignedStg2);
TILING_DATA_FIELD_DEF(int64_t, aOuterStg2);
TILING_DATA_FIELD_DEF(int64_t, aTailStg2);
TILING_DATA_FIELD_DEF(int64_t, binAddBasicBlockLoop);
TILING_DATA_FIELD_DEF(int64_t, binAddMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, binAddCacheBufferCount);
TILING_DATA_FIELD_DEF(int64_t, binAddResultCacheID);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBinAddBasicBlockLoop);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBinAddMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBinAddResultCacheID);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormGrad_1100, BatchNormGradRARSplitCoreR0TilingData);

struct BatchNormGradCompileInfo {
    int32_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    int64_t vlFp32;
};

class BatchNormGradTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormGradTilingBase(gert::TilingContext* context, BatchNormGradBaseTilingData& baseTiling)
        : Ops::NN::Optiling::TilingBaseClass(context), baseTilingData(baseTiling)
    {}

protected:
    // 获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // UB内二分累加参数计算
    void DoBinaryAddTiling(BatchNormGradBinaryAddTilingData& tilingData, int64_t quotient);

    const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape);

    ge::graphStatus CheckInputDtypeValid();
    ge::graphStatus CheckSmallShapesValid();
    ge::graphStatus CheckBigShapesValid();
    ge::graphStatus GetShapesAndCheckValid();
    ge::graphStatus GetDtypesAndCheckValid();

    uint32_t coreNum{0};
    uint64_t ubSize{0};
    uint64_t blockSize{0};
    uint64_t vlFp32{0};

    ge::Format dyFormat;
    ge::DataType dyDtype{ge::DataType::DT_FLOAT};
    ge::DataType weightDtype{ge::DataType::DT_FLOAT};
    int64_t r1Dim{0};          // R1AR0 R外轴
    int64_t aDim{0};           // R1AR0 A轴
    int64_t r0Dim{0};          // R1AR0 R0内轴
    int64_t rAlign{0};         // 用于随路转置为AR，R轴对齐
    int64_t onceProcUbNeed{0}; // A轴上执行一次，需要占用的ub大小
    BatchNormGradBaseTilingData& baseTilingData;
};

class BatchNormGradRARFullLoadTilingBase : public BatchNormGradTilingBase
{
public:
    explicit BatchNormGradRARFullLoadTilingBase(gert::TilingContext* context)
        : BatchNormGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    int64_t reservUbSizeForAlign{0};
    int64_t binaryAddUbNeed{0};
    int64_t binaryAddQuotient{0};
    BatchNormGradRARFullLoadTilingData tilingData;
};

class BatchNormGradRARRecomputeTilingBase : public BatchNormGradTilingBase
{
public:
    explicit BatchNormGradRARRecomputeTilingBase(gert::TilingContext* context)
        : BatchNormGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void DoRecomputeTilingSplitR1();
    void DoRecomputeTilingSplitR0();

    uint64_t binaryAddBufSize{0};
    uint64_t subTilingKey{0};
    int64_t r1Factor{0};
    int64_t r0Factor{0};
    int64_t ubRDimLoopNum{0};
    int64_t ubRDimFactor{0};
    int64_t ubRDimFactorAlign{0};
    int64_t ubRDimTailFactor{0};
    int64_t ubRDimTailFactorAlign{0};
    int64_t ubRDimTail{0};
    int64_t ubRDimTailLoopNum{0};
    int64_t ubRDimTailTail{0};
    int64_t ubRDimTailTailFactor{0};
    int64_t ubRDimTailTailFactorAlign{0};
    int64_t ubRDimTailTailLoopNum{0};
    BatchNormGradRARRecomputeTilingData tilingData;
};

class BatchNormGradRAFullLoadTilingBase : public BatchNormGradTilingBase
{
public:
    explicit BatchNormGradRAFullLoadTilingBase(gert::TilingContext* context)
        : BatchNormGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void SetBlockFactors(int64_t aDim_, int64_t dtypeSize);
    void CalculateLoopFactors(int64_t dtypeSize, int64_t weightDtypeSize, int64_t rDim_, int64_t power2k);
    void SetReduceLoopTimes(int64_t power2k, int64_t rDim_);

private:
    int64_t reservUbSizeForAlign{0};
    int64_t binaryAddUbNeed{0};
    int64_t binaryAddQuotient{0};
    BatchNormGradRAFullLoadTilingData tilingData;
};

class BatchNormGradRARecomputeTilingBase : public BatchNormGradTilingBase
{
public:
    explicit BatchNormGradRARecomputeTilingBase(gert::TilingContext* context)
        : BatchNormGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void SetBlockFactors(int64_t aDim_, int64_t dtypeSize);

private:
    int64_t reservUbSizeForAlign{0};
    int64_t binaryAddUbNeed{0};
    int64_t binaryAddQuotient{0};
    BatchNormGradRARecomputeTilingData tilingData;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_BATCH_NORM_GRAD_H_
