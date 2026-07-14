/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file huber_loss_grad_tiling.cpp
 * \brief HuberLossGrad算子的tiling(分块)策略实现
 *
 * 本文件提供tiling逻辑，将计算任务划分为较小的块在AI Core上并行执行。
 * 采用基于UB容量约束的二分搜索求解最优tile长度。
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h" // GetSignMaxMinTmpSize: host侧获取Sign高阶接口所需临时空间范围
#include "../op_kernel/huber_loss_grad_tiling_data.h"
#include "../op_kernel/huber_loss_grad_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0U;

// Kernel侧缓冲区布局:
//   TQue (VECIN/VECOUT, 双缓冲): predictionsQueue, targetsQueue, gradOutputQueue -> 3个queue * BUFFER_NUM
//   TBuf (VECCALC, sizeof(T)): diffBuf, absBuf, signBuf, gradLargeBuf -> 4个
//   TBuf (VECCALC, sizeof(uint8_t)): maskBuf -> 1个
//   TBuf (VECCALC, GetSignMaxMinTmpSize): signTmpBuf -> 1个（大小由Sign高阶接口确定）
constexpr uint32_t NUM_QUEUES = 3;
constexpr uint32_t NUM_TMP_T = 4;
constexpr uint32_t NUM_TMP_U8 = 1;

// bf16特化 UB 布局: Queue用bf16(2B), 临时缓冲用float(4B)+uint8_t
//   signTmpBuf在bf16特化下作用于float，大小由GetSignMaxMinTmpSize确定（已并入float估算）
constexpr uint32_t NUM_TMP_FLOAT_BF16 = 7;
constexpr uint32_t NUM_TMP_U8_BF16 = 1;

// 通过Sign高阶接口获取sharedTmpBuffer所需临时空间。
// 在[minValue, maxValue]范围内取maxValue，以获得更优的Sign计算性能（符合检视意见的优化意图）；
// 该值会参与CalcUbUsage的UB约束求解，自动保证总占用不超UB（tileDataNum相应自适应）。
static uint32_t CalcSignTmpBufSize(uint32_t tileLen, uint32_t typeBytes)
{
    uint32_t signMaxValue = 0;
    uint32_t signMinValue = 0;
    ge::Shape signShape(std::vector<int64_t>{static_cast<int64_t>(tileLen)});
    AscendC::GetSignMaxMinTmpSize(signShape, typeBytes, false, signMaxValue, signMinValue);
    // 取maxValue并对齐到BLOCK_SIZE(32B)，满足UB分配粒度
    return ((signMaxValue + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
}

static uint32_t CalcUbUsage(uint32_t tileLen, uint32_t typeBytes)
{
    uint32_t alignT = ((tileLen * typeBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t alignU8 = ((tileLen + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    uint32_t total = 0;
    total += NUM_QUEUES * BUFFER_NUM * alignT;
    total += NUM_TMP_T * alignT;
    total += CalcSignTmpBufSize(tileLen, typeBytes); // signTmpBuf精确占用
    total += NUM_TMP_U8 * alignU8;
    return total;
}

static uint32_t FindOptimalTileLength(uint64_t ubSize, uint32_t typeBytes)
{
    constexpr uint32_t alignUnit = 32;

    uint32_t lo = alignUnit;
    // hi上界：signTmpBuf近似为1个alignT参与估算
    uint32_t hi = ubSize / ((NUM_QUEUES * BUFFER_NUM + NUM_TMP_T + 1) * typeBytes + NUM_TMP_U8);
    hi = (hi / alignUnit) * alignUnit;
    if (hi < lo)
        hi = lo;

    uint32_t best = lo;
    while (lo <= hi) {
        uint32_t mid = ((lo + hi) / 2 / alignUnit) * alignUnit;
        if (mid < lo)
            mid = lo;

        if (CalcUbUsage(mid, typeBytes) <= ubSize) {
            best = mid;
            lo = mid + alignUnit;
        } else {
            if (mid <= lo)
                break;
            hi = mid - alignUnit;
        }
    }
    return best;
}

static uint32_t CalcUbUsageBf16(uint32_t tileLen)
{
    uint32_t alignBf16 = ((tileLen * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t alignFloat = ((tileLen * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t alignU8 = ((tileLen + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    uint32_t total = 0;
    total += NUM_QUEUES * BUFFER_NUM * alignBf16;
    total += NUM_TMP_FLOAT_BF16 * alignFloat;
    total += CalcSignTmpBufSize(tileLen, sizeof(float)); // bf16特化Sign在float上
    total += NUM_TMP_U8_BF16 * alignU8;
    return total;
}

static uint32_t FindOptimalTileLengthBf16(uint64_t ubSize)
{
    constexpr uint32_t alignUnit = 32;

    uint32_t lo = alignUnit;
    // hi上界：signTmpBuf近似为1个alignFloat(4B)参与估算
    uint32_t hi = ubSize / (NUM_QUEUES * BUFFER_NUM * 2 + (NUM_TMP_FLOAT_BF16 + 1) * 4 + NUM_TMP_U8_BF16);
    hi = (hi / alignUnit) * alignUnit;
    if (hi < lo)
        hi = lo;

    uint32_t best = lo;
    while (lo <= hi) {
        uint32_t mid = ((lo + hi) / 2 / alignUnit) * alignUnit;
        if (mid < lo)
            mid = lo;

        if (CalcUbUsageBf16(mid) <= ubSize) {
            best = mid;
            lo = mid + alignUnit;
        } else {
            if (mid <= lo)
                break;
            hi = mid - alignUnit;
        }
    }
    return best;
}

struct HuberLossGradCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HuberLossGradTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行时信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取输入信息
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dt, typeLength);

    // 3. 获取WorkspaceSize
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4. 求解最优tile长度
    uint32_t tileDataNum;
    if (dt == ge::DT_BF16) {
        tileDataNum = FindOptimalTileLengthBf16(ubSize);
    } else {
        tileDataNum = FindOptimalTileLength(ubSize, typeLength);
    }
    uint32_t tileBlockNum = (tileDataNum * typeLength) / BLOCK_SIZE;

    // 5. 核间负载均衡
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;

    {
        uint32_t minBlocksPerCore = tileBlockNum * 2;
        uint32_t maxEffectiveCores = (inputLengthAlgin32 / BLOCK_SIZE) / minBlocksPerCore;
        if (maxEffectiveCores < 1)
            maxEffectiveCores = 1;
        if (coreNum > maxEffectiveCores) {
            coreNum = maxEffectiveCores;
        }
    }

    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 6. 获取delta属性值
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* deltaPoint = attrs->GetAttrPointer<float>(0);
    const float delta = (deltaPoint != nullptr) ? *deltaPoint : 1.0f;

    // 7. 设置tiling数据
    HuberLossGradTilingData* tiling = context->GetTilingData<HuberLossGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(memset_s(tiling, sizeof(HuberLossGradTilingData), 0, sizeof(HuberLossGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->inputNum = inputNum;
    tiling->dataTypeId = dt;
    tiling->delta = delta;

    // Sign高阶接口临时空间：bf16特化下Sign在float上计算，其余按输入typeLength
    uint32_t signTypeBytes = (dt == ge::DT_BF16) ? sizeof(float) : typeLength;
    tiling->signTmpSize = CalcSignTmpBufSize(tileDataNum, signTypeBytes);

    context->SetBlockDim(coreNum);

    // 数据类型合法性校验，避免非法 dtype 触发后续除零等异常。
    if (dt != ge::DT_FLOAT && dt != ge::DT_FLOAT16 && dt != ge::DT_BF16) {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(HUBER_LOSS_GRAD_TPL_SCH_MODE_0));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHuberLossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HuberLossGrad)
    .Tiling(HuberLossGradTilingFunc)
    .TilingParse<HuberLossGradCompileInfo>(TilingParseForHuberLossGrad);
} // namespace optiling
