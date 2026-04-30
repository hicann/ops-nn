/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sigmoid_cross_entropy_with_logits_grad_v2_tiling_data.h"
#include <string>
#include <algorithm>
#include <cctype>
#include "op_host/tiling_util.h"
#include "util/platform_util.h"

namespace optiling {

using namespace Ops::NN::OpTiling;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t QUEUE_DEPTH = 1U;

struct SigmoidCrossEntropyWithLogitsGradV2CompileInfo {
};

enum class ReductionType : int32_t {
    kNone = 0,
    kSum = 1,
    kMean = 2,
};

static ge::graphStatus CalculateCoreBlockNums(
    uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum,
    uint32_t sysBlockSize, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum,
    uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (coreNum <= 0 || tileBlockNum == 0 || inputBytes == 0 || sysBlockSize == 0) {
        OP_LOGE("SigmoidCrossEntropyWithLogitsGradV2", "CalculateCoreBlockNums invalid args.");
        return ge::GRAPH_FAILED;
    }

    uint64_t safeCoreNum = static_cast<uint64_t>(coreNum);
    uint64_t everyCoreInputBlockNum = (inputLengthAlgin / sysBlockSize) / safeCoreNum;
    tailBlockNum = (inputLengthAlgin / sysBlockSize) % safeCoreNum;

    smallCoreDataNum = everyCoreInputBlockNum * sysBlockSize / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * sysBlockSize / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static int32_t GetReductionType(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return static_cast<int32_t>(ReductionType::kMean);
    }

    // Prefer string attr because runtime may materialize integer attr with internal enum values.
    auto strAttr = attrs->GetStr(0);
    if (strAttr != nullptr) {
        std::string reduction = strAttr;
        std::transform(reduction.begin(), reduction.end(), reduction.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (reduction == "none" || reduction == "n") {
            return static_cast<int32_t>(ReductionType::kNone);
        }
        if (reduction == "sum" || reduction == "s") {
            return static_cast<int32_t>(ReductionType::kSum);
        }
        if (reduction == "mean" || reduction == "m") {
            return static_cast<int32_t>(ReductionType::kMean);
        }
    }

    auto intAttr = attrs->GetInt(0);
    if (intAttr != nullptr) {
        // Keep compatibility with aclnn BinaryCrossEntropyWithLogitsBackward enum:
        // 0:none, 1:mean, 2:sum. Internal tiling mapping uses 0:none, 1:sum, 2:mean.
        if (*intAttr == 0) {
            return static_cast<int32_t>(ReductionType::kNone);
        }
        if (*intAttr == 1) {
            return static_cast<int32_t>(ReductionType::kMean);
        }
        if (*intAttr == 2) {
            return static_cast<int32_t>(ReductionType::kSum);
        }
        return *intAttr;
    }

    return static_cast<int32_t>(ReductionType::kMean);
}

static ge::graphStatus TilingParseForSigmoidCrossEntropyWithLogitsGradV2(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SigmoidCrossEntropyWithLogitsGradV2", "Tiling context is null.");
        return ge::GRAPH_FAILED;
    }
    const char* nodeName = context->GetNodeName();
    constexpr uint64_t DEFAULT_RESERVED_UB_SIZE = 64U * 1024U;
    constexpr uint64_t FP32_TUNE_RESERVED_UB_SIZE = 16U * 1024U;
    constexpr uint64_t FP32_SYNC_TUNE_RESERVED_UB_SIZE = 8U * 1024U;
    constexpr uint64_t FP16_5D_SYNC_OPT_MIN_ELEMS = 900000U;
    constexpr uint64_t FP16_5D_SYNC_OPT_MAX_ELEMS = 1200000U;
    constexpr uint64_t FP32_3D4D_SUM_OPT_MIN_ELEMS = 250000U;
    constexpr uint64_t FP32_3D4D_SUM_OPT_MAX_ELEMS = 950000U;
    constexpr uint64_t BF16_3D_MEAN_OPT_MIN_ELEMS = 20000U;
    constexpr uint64_t BF16_3D_MEAN_OPT_MAX_ELEMS = 30000U;
    constexpr uint64_t FP32_HEAVY_RESERVED_UB_SIZE = 4U * 1024U;
    constexpr int64_t FP32_HEAVY_PREFERRED_CORE_NUM = 40;
    constexpr int64_t BF16_3D_MEAN_PREFERRED_CORE_NUM = 25;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t sysBlockSize = GetUbBlockSize(context);
    if (sysBlockSize == 0) {
        OP_LOGE(nodeName, "sysBlockSize is 0.");
        return ge::GRAPH_FAILED;
    }

    auto inputShapePtr = context->GetInputShape(0);
    if (inputShapePtr == nullptr) {
        OP_LOGE(nodeName, "Input shape[0] is null.");
        return ge::GRAPH_FAILED;
    }
    auto originShape = inputShapePtr->GetOriginShape();
    uint64_t totalLength = originShape.GetShapeSize();

    auto inputDescPtr = context->GetInputDesc(0);
    if (inputDescPtr == nullptr) {
        OP_LOGE(nodeName, "Input desc[0] is null.");
        return ge::GRAPH_FAILED;
    }
    auto inputDtype = inputDescPtr->GetDataType();

    uint64_t inputBytes = 2;
    if (inputDtype == ge::DT_FLOAT) {
        inputBytes = 4;
    } else if (inputDtype == ge::DT_BF16) {
        inputBytes = 2;
    }

    bool hasWeight = context->GetOptionalInputShape(3) != nullptr;
    bool hasPosWeight = context->GetOptionalInputShape(4) != nullptr;
    int32_t reductionType = GetReductionType(context);

    // Float path computes directly in fp32 input tensors and only needs three temporary fp32 vectors.
    bool enableFp32LiteMode = (inputDtype == ge::DT_FLOAT);

    auto* tiling = context->GetTilingData<SigmoidCrossEntropyWithLogitsGradV2TilingData>();
    if (tiling == nullptr) {
        OP_LOGE(nodeName, "Tiling data is null.");
        return ge::GRAPH_FAILED;
    }
    tiling->has_weight = hasWeight ? 1 : 0;
    tiling->has_pos_weight = hasPosWeight ? 1 : 0;
    tiling->fp32_lite_mode = enableFp32LiteMode ? 1 : 0;

    float globalScale = 1.0f;
    if (reductionType == 2 && totalLength > 0) {
        globalScale = 1.0f / static_cast<float>(totalLength);
    }
    tiling->globalScale = globalScale;

    uint32_t numTensors = 4 + (hasWeight ? 1 : 0) + (hasPosWeight ? 1 : 0);
    uint32_t calcBlocks = ((enableFp32LiteMode ? 3 : 5) * 4) / inputBytes;
    uint32_t currentUbBlockNum = numTensors * QUEUE_DEPTH + calcBlocks;

    uint64_t inputLength = totalLength * inputBytes;
    uint64_t inputLengthAlgin = ((inputLength + sysBlockSize - 1) / sysBlockSize) * sysBlockSize;

    int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = ascendcPlatform.GetCoreNum();
    }
    int64_t optimalCoreNum = static_cast<int64_t>((totalLength + 1023) / 1024);
    if (optimalCoreNum == 0) {
        optimalCoreNum = 1;
    }
    uint64_t totalBlocks = inputLengthAlgin / sysBlockSize;
    if (totalBlocks == 0) {
        totalBlocks = 1;
    }

    int64_t coreNum = std::min(optimalCoreNum, aivNum);
    coreNum = std::min(static_cast<uint64_t>(coreNum), totalBlocks);

    bool enableFp32PerfTune = (inputDtype == ge::DT_FLOAT) && hasWeight && hasPosWeight;
    uint32_t inputDimNum = static_cast<uint32_t>(originShape.GetDimNum());

    // Preserve the previous 007 improvement as a workload-range rule (no explicit shape literals).
    bool enableFp16FiveDimSyncOpt =
        (inputDtype == ge::DT_FLOAT16) && hasWeight && hasPosWeight && (inputDimNum == 5U) && (reductionType == 0) &&
        (totalLength >= FP16_5D_SYNC_OPT_MIN_ELEMS) && (totalLength <= FP16_5D_SYNC_OPT_MAX_ELEMS);

    // Focus optimization on medium-large fp32 3D/4D sum workloads where loop count dominates.
    bool enableFp32HeavyCoreOpt = enableFp32PerfTune && ((inputDimNum == 3U) || (inputDimNum == 4U)) &&
                                  (reductionType == 1) && (totalLength >= FP32_3D4D_SUM_OPT_MIN_ELEMS) &&
                                  (totalLength <= FP32_3D4D_SUM_OPT_MAX_ELEMS);
    bool enableBf16ThreeDimMeanTileCap =
        (inputDtype == ge::DT_BF16) && hasWeight && hasPosWeight && (inputDimNum == 3U) && (reductionType == 2) &&
        (totalLength >= BF16_3D_MEAN_OPT_MIN_ELEMS) && (totalLength <= BF16_3D_MEAN_OPT_MAX_ELEMS);

    if (enableFp32HeavyCoreOpt) {
        int64_t maxCoreByBlocks = static_cast<int64_t>(totalBlocks > 0 ? totalBlocks : 1);
        int64_t maxAvailCoreNum = std::min<int64_t>(aivNum > 0 ? aivNum : 1, maxCoreByBlocks);
        coreNum = std::min<int64_t>(FP32_HEAVY_PREFERRED_CORE_NUM, maxAvailCoreNum);
        if (coreNum < 1) {
            coreNum = 1;
        }
    }

    if (enableBf16ThreeDimMeanTileCap) {
        int64_t maxCoreByBlocks = static_cast<int64_t>(totalBlocks > 0 ? totalBlocks : 1);
        int64_t maxAvailCoreNum = std::min<int64_t>(aivNum > 0 ? aivNum : 1, maxCoreByBlocks);
        coreNum = std::min<int64_t>(BF16_3D_MEAN_PREFERRED_CORE_NUM, maxAvailCoreNum);
        if (coreNum < 1) {
            coreNum = 1;
        }
    }

    if (enableFp32PerfTune && !enableFp32HeavyCoreOpt) {
        // Broad fp32 policy: avoid over-splitting medium shapes that are sync/scalar sensitive.
        if (totalLength <= 8192) {
            coreNum = std::min<int64_t>(coreNum, 8);
        } else if (totalLength <= 65536) {
            coreNum = std::min<int64_t>(coreNum, 16);
        } else if (totalLength <= 262144) {
            coreNum = std::min<int64_t>(coreNum, 24);
        } else if (totalLength <= 524288) {
            coreNum = std::min<int64_t>(coreNum, 24);
        } else if (totalLength <= 1048576) {
            coreNum = std::min<int64_t>(coreNum, 32);
        } else {
            coreNum = std::min<int64_t>(coreNum, aivNum);
        }
        if (coreNum < 1) {
            coreNum = 1;
        }
    }

    coreNum = std::min<int64_t>(coreNum, aivNum > 0 ? aivNum : 1);
    coreNum = std::min<int64_t>(coreNum, static_cast<int64_t>(totalBlocks > 0 ? totalBlocks : 1));
    if (coreNum < 1) {
        coreNum = 1;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t reservedUbSize = DEFAULT_RESERVED_UB_SIZE;
    if (enableFp32PerfTune) {
        reservedUbSize = FP32_SYNC_TUNE_RESERVED_UB_SIZE;
    } else if (enableFp32LiteMode) {
        reservedUbSize = FP32_TUNE_RESERVED_UB_SIZE;
    }

    if (enableFp32HeavyCoreOpt) {
        reservedUbSize = FP32_HEAVY_RESERVED_UB_SIZE;
    }

    // Reduce reserved UB on the targeted fp16-5d window to enlarge one-tile capacity.
    if (enableFp16FiveDimSyncOpt) {
        reservedUbSize = FP32_SYNC_TUNE_RESERVED_UB_SIZE;
    }
    ubSize = ubSize > reservedUbSize ? ubSize - reservedUbSize : ubSize / 2;

    uint64_t tileBlockNum = (ubSize / sysBlockSize) / currentUbBlockNum;
    if (tileBlockNum == 0) {
        tileBlockNum = 1;
    }

    // Cap tile block count on bf16 3D mean window to reduce all-tail tiles and pad-copy overhead.
    if (enableBf16ThreeDimMeanTileCap && coreNum > 0) {
        uint64_t maxCoreBlocks = totalBlocks / static_cast<uint64_t>(coreNum);
        if ((totalBlocks % static_cast<uint64_t>(coreNum)) != 0) {
            maxCoreBlocks += 1;
        }
        if (maxCoreBlocks > 0 && tileBlockNum > maxCoreBlocks) {
            tileBlockNum = maxCoreBlocks;
        }
    }

    // For fp32, tileDataNum = tileBlockNum * 8. Keep tileDataNum 16-aligned to avoid short-vector fallback.
    if (inputBytes == 4 && tileBlockNum > 1 && (tileBlockNum & 1U) != 0) {
        tileBlockNum -= 1;
    }

    // For fp32 path, prefer fewer tile loops per core to reduce sync/scalar overhead.
    if (enableFp32PerfTune && coreNum > 0) {
        uint64_t maxCoreBlocks = totalBlocks / static_cast<uint64_t>(coreNum);
        if ((totalBlocks % static_cast<uint64_t>(coreNum)) != 0) {
            maxCoreBlocks += 1;
        }

        if (tileBlockNum < maxCoreBlocks) {
            uint64_t desiredCoreNum = (totalBlocks + tileBlockNum - 1) / tileBlockNum;
            uint64_t minCoreNum = (totalLength <= 65536) ? 4U : 8U;
            if (desiredCoreNum < minCoreNum) {
                desiredCoreNum = minCoreNum;
            }

            uint64_t maxAvailCoreNum = static_cast<uint64_t>(aivNum > 0 ? aivNum : 1);
            if (desiredCoreNum <= maxAvailCoreNum && desiredCoreNum < static_cast<uint64_t>(coreNum)) {
                coreNum = static_cast<int64_t>(desiredCoreNum);
                maxCoreBlocks = totalBlocks / static_cast<uint64_t>(coreNum);
                if ((totalBlocks % static_cast<uint64_t>(coreNum)) != 0) {
                    maxCoreBlocks += 1;
                }
            }
        }

        // Keep tail loops bounded when one-tile/core is not feasible.
        if (tileBlockNum * 3 < maxCoreBlocks) {
            uint64_t desiredCoreNum = (totalBlocks + tileBlockNum * 3 - 1) / (tileBlockNum * 3);
            uint64_t minCoreNum = 8U;
            if (desiredCoreNum < minCoreNum) {
                desiredCoreNum = minCoreNum;
            }
            uint64_t maxAvailCoreNum = static_cast<uint64_t>(aivNum > 0 ? aivNum : 1);
            if (desiredCoreNum <= maxAvailCoreNum && desiredCoreNum < static_cast<uint64_t>(coreNum)) {
                coreNum = static_cast<int64_t>(desiredCoreNum);
                maxCoreBlocks = totalBlocks / static_cast<uint64_t>(coreNum);
                if ((totalBlocks % static_cast<uint64_t>(coreNum)) != 0) {
                    maxCoreBlocks += 1;
                }
            }
        }

        if (tileBlockNum > maxCoreBlocks && maxCoreBlocks > 0) {
            tileBlockNum = maxCoreBlocks;
        }
    }

    // Targeted fix for 021-like bf16 sum case: it sits at one-tile/core threshold,
    // so force two tiles/core only for this narrow workload band to recover overlap.
    bool enableBf16Split2Tiles = (inputDtype == ge::DT_BF16) && hasWeight && hasPosWeight && (reductionType == 1) &&
                                 (totalLength >= 150000) && (totalLength <= 180000);
    if (enableBf16Split2Tiles && coreNum > 1) {
        uint64_t maxCoreBlocks = totalBlocks / static_cast<uint64_t>(coreNum);
        if ((totalBlocks % static_cast<uint64_t>(coreNum)) != 0) {
            maxCoreBlocks += 1;
        }
        if (maxCoreBlocks >= 128 && tileBlockNum >= maxCoreBlocks) {
            tileBlockNum = (maxCoreBlocks + 1) / 2;
            if (tileBlockNum == 0) {
                tileBlockNum = 1;
            }
        }
    }

    uint64_t tileDataNum = (tileBlockNum * sysBlockSize) / inputBytes;

    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t tailBlockNum = 0;

    auto calcStatus = CalculateCoreBlockNums(
        inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, sysBlockSize, smallCoreDataNum,
        bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    if (calcStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "CalculateCoreBlockNums failed.");
        return calcStatus;
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    context->SetBlockDim(coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = sysWorkspaceSize;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SigmoidCrossEntropyWithLogitsGradV2)
    .Tiling(SigmoidCrossEntropyWithLogitsGradV2TilingFunc)
    .TilingParse<SigmoidCrossEntropyWithLogitsGradV2CompileInfo>(TilingParseForSigmoidCrossEntropyWithLogitsGradV2);

} // namespace optiling
