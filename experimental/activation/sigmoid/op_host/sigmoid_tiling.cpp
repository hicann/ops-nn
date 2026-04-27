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
 * \file sigmoid_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

uint32_t blockSize;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;

constexpr uint64_t THRESHOLD_4K = 4096;
constexpr uint64_t THRESHOLD_16K = 16384;
constexpr uint64_t THRESHOLD_64K = 65536;
constexpr uint64_t THRESHOLD_128K = 131072;
constexpr uint64_t THRESHOLD_512K = 524288;
constexpr uint64_t DATA_PER_CORE = 1024;

constexpr uint64_t CORES_TIER1 = 4;
constexpr uint64_t CORES_TIER2 = 8;
constexpr uint64_t CORES_TIER3 = 16;

// UB Data Number Constants
constexpr uint64_t UB_DATA_NUM_HIGH_PERF_310P_BF16 = 10;
constexpr uint64_t UB_DATA_NUM_HIGH_PERF_310P_OTHER = 6;
constexpr uint64_t UB_DATA_NUM_CAST_TO_FLOAT = 8;
constexpr uint64_t UB_DATA_NUM_NATIVE = 5;

struct SigmoidCompileInfo {};

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize <= 0, OP_LOGE(context, "blockSize is less than or equal to 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is less than or equal to 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    
    // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); 
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(context, "currentWorkspace is nullptr"), return ge::GRAPH_FAILED);
    
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetUbDataNum(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    auto dataType = context->GetInputDesc(0)->GetDataType();

    uint64_t ubDataNumber = 0;
    bool isHighPerf = false;
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    isHighPerf = true;
#endif
    bool isBf16 = (dataType == ge::DT_BF16); 
    bool isFp16 = (dataType == ge::DT_FLOAT16);

    if (isHighPerf && socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        // --- 310P 高性能模式 (Poly) ---
        if (isBf16) {
            // In(2) + Out(2) + Poly1(2*2B) + Poly2(2*2B) + Cast(2*2B) = 4 + 2 + 2 + 2 = 10
            ubDataNumber = UB_DATA_NUM_HIGH_PERF_310P_BF16;
        } else {
            // FP16/FP32: In(2) + Out(2) + Poly1(1) + Poly2(1) = 6
            ubDataNumber = UB_DATA_NUM_HIGH_PERF_310P_OTHER;
        }
    } else {
        // --- 常规计算模式 ---
        // 判断是否需要 Cast 到 Float (BF16 必须，FP16 在特定情况需要)
        bool needCastToFloat = isBf16;
        if (!isHighPerf && isFp16 && 
            (socVersion == platform_ascendc::SocVersion::ASCEND310P || 
             socVersion == platform_ascendc::SocVersion::ASCEND910 || 
             socVersion == platform_ascendc::SocVersion::ASCEND910B || 
             socVersion == platform_ascendc::SocVersion::ASCEND910_93)) {
            needCastToFloat = true;
        }

        if (needCastToFloat) {
            // In(2) + Out(2) + CastBuffer(2*float/dtype) + OnesBuffer(2*float/dtype)
            // 对于 BF16/FP16，float 是 2 倍大小，所以是 2+2+2+2 = 8
            ubDataNumber = UB_DATA_NUM_CAST_TO_FLOAT; 
        } else {
            // Native FP32 or Native FP16
            // In(2) + Out(2) + Ones(1) = 5
            ubDataNumber = UB_DATA_NUM_NATIVE;
        }
    }
    return ubDataNumber;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlgin32)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputShape(0) == nullptr, OP_LOGE(context, "InputShape is nullptr"), return ge::GRAPH_FAILED);
    
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is 0");
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    uint64_t ubDataNumber = GetUbDataNum(context);
    if(ubDataNumber == 0) {
        OP_LOGE(context, "ubDataNumber is 0");
        return ge::GRAPH_FAILED;
    }
    if (blockSize == 0) {
        OP_LOGE(context, "blockSize is 0");
        return ge::GRAPH_FAILED;
    }
    tileBlockNum = (ubSize / blockSize) / ubDataNumber;
    
    if (inputBytes == 0) {
        OP_LOGE(context, "inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    
    tileDataNum = (tileBlockNum * blockSize) / inputBytes;
    inputLengthAlgin32 = (((inputLength + blockSize - 1) / blockSize) * blockSize);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(gert::TilingContext* context,
    uint64_t inputLengthAlgin32, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum,
    uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum,
    uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (blockSize == 0 || coreNum <= 0 || tileBlockNum == 0 || inputBytes == 0) {
        OP_LOGE(context, "invalid blockSize/coreNum/tileBlockNum/inputBytes");
        return ge::GRAPH_FAILED;
    }
    
    uint64_t coreNumUint = static_cast<uint64_t>(coreNum);
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / blockSize / coreNumUint;
    tailBlockNum = (inputLengthAlgin32 / blockSize) % coreNumUint;
    
    smallCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static uint64_t LimitCoreNum(int64_t maxCoreNum, uint64_t inputNum) 
{
    if (inputNum < THRESHOLD_4K) {
        // 每1024个数据使用1个核心，至少使用1核
        uint64_t cores = (inputNum + DATA_PER_CORE - 1) / DATA_PER_CORE;
        return cores > 0 ? cores : 1;
    } else if (inputNum < THRESHOLD_16K) {
        return CORES_TIER1;
    } else if (inputNum < THRESHOLD_64K) {
        return CORES_TIER2;
    } else if (inputNum < THRESHOLD_512K) {
        return CORES_TIER3;
    } else {
        return maxCoreNum;
    }
}



static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SigmoidTilingData), 0, sizeof(SigmoidTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    
    // 获取平台运行信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    
    // 获取输入数据信息
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    
    // 用限制之后的核数当最大核数进行计算
    uint64_t limitedCoreNum = LimitCoreNum(coreNum, inputNum);
    
    // 计算coreNum
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. 
        // The maximum number of audits is the actual number of audits
        if (blockSize == 0) {
            OP_LOGE(context, "blockSize is 0");
            return ge::GRAPH_FAILED;
        }
        uint64_t maxBlocks = inputLengthAlgin32 / blockSize;
        coreNum = (limitedCoreNum < maxBlocks) ? static_cast<int64_t>(limitedCoreNum) : static_cast<int64_t>(maxBlocks);
    }
    
    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    
    ret = CalculateCoreBlockNums(context,
        inputLengthAlgin32, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);
    
    // 设置tiling数据
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    
    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
        
    uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);
} // namespace optiling