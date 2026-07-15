/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_mx_quant_optimize_tiling.cpp
 * \brief
 */
#include "dynamic_mx_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/dynamic_mx_quant_tilingdata.h"
#include <cmath>
#include "platform/platform_info.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicMxQuant;

namespace optiling {
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr float DST_TYPE_DEFAULT = 0.0;
constexpr float NUM_SIX_FLOAT = 6.0;
constexpr float NUM_SEVEN_FLOAT = 7.0;
constexpr int64_t NUM_TEN = 10;
constexpr int64_t MODE_ZERO = 0;
constexpr int64_t MODE_ONE = 1;
constexpr int64_t MODE_TWO = 2;
constexpr int64_t MODE_THREE = 3;
constexpr int64_t DIGIT_TEN_THOUSAND = 10000;
constexpr int64_t BLOCK_PER_GROUP = 2;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t BYTES_OF_INPUT_TYPE_FP32 = 4;

ge::graphStatus DynamicMxQuantOptimzieTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicMxQuantOptimzieTiling DoTiling.");

    OP_CHECK_IF(SetTilingParam() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "DynamicMxQuantOptimzieTiling SetTilingParam Failed"),
                return ge::GRAPH_FAILED);
    // 设置计算模式，以此区分tilingkey与scale计算方式
    OP_CHECK_IF(SetCalcMode() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Set calculation mode Failed"),
                return ge::GRAPH_FAILED);
    SetTilingKey();
    tilingData_ = context_->GetTilingData<DynamicMxQuant4OptimizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    SetTilingData();
    PrintTilingData();

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", tilingParam_.usedCoreNum);
    context_->SetBlockDim(tilingParam_.usedCoreNum);
    context_->SetTilingKey(tilingParam_.tilingKey);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParam_.workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::SetTilingParam()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicMxQuantOptimzieTiling SetTilingParam.");

    OP_CHECK_IF(tilingParam_.groupPerUb == 0,
                OP_LOGE(context_->GetNodeName(), "Shape too large to fit the optimal template."),
                return ge::GRAPH_FAILED);
    SplitCore();

    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantOptimzieTiling::SplitCore()
{
    // 量化轴对齐blockSize后元素个数, 量化轴包含的block块个数
    tilingParam_.mAlignSize = Ops::Base::CeilAlign(tilingParam_.quantAxisSize, tilingParam_.blockSize);
    tilingParam_.mAlignBlockCount = Ops::Base::CeilDiv(tilingParam_.quantAxisSize, tilingParam_.blockSize);
    // 尾轴对齐之后的元素个数
    tilingParam_.nAlignSize = Ops::Base::CeilAlign(tilingParam_.postAxisSize, tilingParam_.nAlignNum);
    tilingParam_.nAlignBlockCount = Ops::Base::CeilDiv(tilingParam_.postAxisSize, tilingParam_.nAlignNum);
    // 量化轴可分的group的个数
    tilingParam_.mAlignGroupCount = Ops::Base::CeilDiv(tilingParam_.quantAxisSize, tilingParam_.blockSize * NUM_TWO);
    /*
       分核逻辑：
       小尾轴：以group为单位分核，group指竖直方向相邻两个block块
       大尾轴：以groupPerUb个块的block的大小为单位分核
    */
    if (tilingParam_.postAxisSize <= tilingParam_.nAlignNum) { // 小尾轴
        tilingParam_.totalGroupNum = tilingParam_.preAxisSize * tilingParam_.mAlignGroupCount *
                                     tilingParam_.nAlignBlockCount;
        tilingParam_.groupPerCore = Ops::Base::CeilDiv(tilingParam_.totalGroupNum, tilingParam_.totalCoreNum);
        tilingParam_.groupPerTail = (tilingParam_.groupPerCore == 0) ?
                                        0 :
                                        tilingParam_.totalGroupNum % tilingParam_.groupPerCore;
        tilingParam_.usedCoreNum = Ops::Base::CeilDiv(tilingParam_.totalGroupNum, tilingParam_.groupPerCore);
        tilingParam_.totalBlockNum = tilingParam_.totalGroupNum * BLOCK_PER_GROUP;

        if (tilingParam_.groupPerUb * NUM_TWO * tilingParam_.usedCoreNum < tilingParam_.totalBlockNum) {
            tilingParam_.blockNumPerTask = tilingParam_.groupPerUb * NUM_TWO;
        } else {
            tilingParam_.blockNumPerTask = Ops::Base::CeilAlign(
                Ops::Base::CeilDiv(tilingParam_.totalBlockNum, tilingParam_.usedCoreNum), BLOCK_PER_GROUP);
        }
        tilingParam_.totalTaskNum = Ops::Base::CeilDiv(tilingParam_.totalBlockNum, tilingParam_.blockNumPerTask);
        // 尾轴是否需要补
        tilingParam_.needPadPostAxis = tilingParam_.postAxisSize % tilingParam_.nAlignNum != 0;
    } else { // 大尾轴
        // fp32 输入按 4 字节计算 ubRowLen，bf16/fp16 按 2 字节
        int64_t inBytes = tilingParam_.isFp32Input ? BYTES_OF_INPUT_TYPE_FP32 : BYTES_OF_INPUT_TYPE;
        tilingParam_.ubRowLen = tilingParam_.vfLen / inBytes;
        tilingParam_.ubRowLenTail = tilingParam_.postAxisSize % tilingParam_.ubRowLen == 0 ?
                                        tilingParam_.ubRowLen :
                                        tilingParam_.postAxisSize % tilingParam_.ubRowLen;
        tilingParam_.ubRowCount = tilingParam_.groupPerUb * NUM_TWO * tilingParam_.blockSize;
        tilingParam_.ubRowCountTail = tilingParam_.quantAxisSize % tilingParam_.ubRowCount == 0 ?
                                          tilingParam_.ubRowCount :
                                          tilingParam_.quantAxisSize % tilingParam_.ubRowCount;
        tilingParam_.nAlignBlockCount = Ops::Base::CeilDiv(tilingParam_.postAxisSize, tilingParam_.ubRowLen);
        tilingParam_.mAlignBlockCount = Ops::Base::CeilDiv(tilingParam_.quantAxisSize, tilingParam_.ubRowCount);
        tilingParam_.blockCountPerBatch = tilingParam_.nAlignBlockCount * tilingParam_.mAlignBlockCount;
        tilingParam_.scaleRowCountPerBatch = Ops::Base::CeilDiv(tilingParam_.quantAxisSize,
                                                                NUM_TWO * tilingParam_.blockSize) *
                                             NUM_TWO;
        tilingParam_.totalTaskNum = tilingParam_.preAxisSize * tilingParam_.blockCountPerBatch;
        tilingParam_.usedCoreNum = std::min(tilingParam_.totalCoreNum, tilingParam_.totalTaskNum);
        tilingParam_.loopNumPerHeadCore = Ops::Base::CeilDiv(tilingParam_.totalTaskNum, tilingParam_.totalCoreNum);
        tilingParam_.loopNumPerTailCore = tilingParam_.totalTaskNum / tilingParam_.totalCoreNum;
    }
    // 量化轴是否需要补block用于交织
    tilingParam_.quantAxisIsOdd = Ops::Base::CeilDiv(tilingParam_.quantAxisSize, tilingParam_.blockSize) % NUM_TWO;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::SetCalcMode()
{
    if (tilingParam_.scaleAlg == NUM_ZERO) {
        tilingParam_.calcMode = MODE_ZERO;
    } else if (tilingParam_.scaleAlg == NUM_ONE) {
        tilingParam_.calcMode = MODE_ONE;
    } else if (tilingParam_.scaleAlg == NUM_TWO) {
        if (Ops::Base::IsFloatEqual(tilingParam_.dstTypeMax, DST_TYPE_DEFAULT) ||
            Ops::Base::IsFloatEqual(tilingParam_.dstTypeMax, NUM_SIX_FLOAT) ||
            Ops::Base::IsFloatEqual(tilingParam_.dstTypeMax, NUM_SEVEN_FLOAT)) {
            tilingParam_.calcMode = MODE_TWO;
        } else {
            tilingParam_.calcMode = MODE_THREE;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale_alg",
                                              std::to_string(tilingParam_.scaleAlg),
                                              "The value of scale_alg must be [0, 1, 2]");
        return ge::GRAPH_FAILED;
    }

    if (tilingParam_.calcMode == MODE_TWO) {
        if (Ops::Base::IsFloatEqual(tilingParam_.dstTypeMax, DST_TYPE_DEFAULT) ||
            Ops::Base::IsFloatEqual(tilingParam_.dstTypeMax, NUM_SIX_FLOAT)) {
            tilingParam_.subNumForScale = 0x000000c1;
        } else {
            tilingParam_.subNumForScale = 0x000000e1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantOptimzieTiling::SetTilingKey()
{
    uint64_t optiMode = tilingParam_.postAxisSize <= tilingParam_.nAlignNum ? TPL_NOT_TAIL_AXIS_QUANT_SMALL_OPTI :
                                                                              TPL_NOT_TAIL_AXIS_QUANT_LARGE_OPTI;
    // 0,1,4分别代表round，floor，rint
    uint64_t roundMode = tilingParam_.roundMode;
    // 对应scaleAlg，可取0,1,2dst_value_max特殊值的情况下是2）,3（dst_value_max非特殊值的情况下是3）
    uint64_t scaleAlg = tilingParam_.calcMode;
    tilingParam_.tilingKey = GET_TPL_TILING_KEY(optiMode, scaleAlg, roundMode, TPL_NOT_CARE_SCALE);
}

ge::graphStatus DynamicMxQuantOptimzieTiling::SetTilingData() const
{
    tilingData_->totalCoreNum = tilingParam_.totalCoreNum;
    tilingData_->usedCoreNum = tilingParam_.usedCoreNum;
    tilingData_->blockSize = tilingParam_.blockSize;
    tilingData_->isPad = tilingParam_.isPad ? 1 : 0;
    tilingData_->tailBlockSize = tilingParam_.tailBlockSize;
    tilingData_->tilingKey = tilingParam_.tilingKey;
    tilingData_->quantAxisSize = tilingParam_.quantAxisSize;
    tilingData_->postAxisSize = tilingParam_.postAxisSize;
    tilingData_->nAlignSize = tilingParam_.nAlignSize;
    tilingData_->mAlignBlockCount = tilingParam_.mAlignBlockCount;
    tilingData_->nAlignBlockCount = tilingParam_.nAlignBlockCount;
    tilingData_->mAlignGroupCount = tilingParam_.mAlignGroupCount;
    tilingData_->quantAxisIsOdd = tilingParam_.quantAxisIsOdd;
    tilingData_->totalGroupNum = tilingParam_.totalGroupNum;
    tilingData_->groupPerUb = tilingParam_.groupPerUb;
    tilingData_->totalBlockNum = tilingParam_.totalBlockNum;
    tilingData_->blockNumPerTask = tilingParam_.blockNumPerTask;
    tilingData_->totalTaskNum = tilingParam_.totalTaskNum;
    tilingData_->loopNumPerHeadCore = tilingParam_.loopNumPerHeadCore;
    tilingData_->loopNumPerTailCore = tilingParam_.loopNumPerTailCore;
    tilingData_->ubRowLen = tilingParam_.ubRowLen;
    tilingData_->ubRowLenTail = tilingParam_.ubRowLenTail;
    tilingData_->ubRowCount = tilingParam_.ubRowCount;
    tilingData_->ubRowCountTail = tilingParam_.ubRowCountTail;
    tilingData_->subNumForScale = tilingParam_.subNumForScale;
    tilingData_->blockCountPerBatch = tilingParam_.blockCountPerBatch;
    tilingData_->scaleRowCountPerBatch = tilingParam_.scaleRowCountPerBatch;
    tilingData_->needPadPostAxis = tilingParam_.needPadPostAxis ? 1 : 0;
    tilingData_->invDstTypeMax = tilingParam_.invDstTypeMax;
    tilingData_->maxLowBound = tilingParam_.maxLowBound;

    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantOptimzieTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData totalCoreNum: %ld, usedCoreNum: %ld, blockSize: %ld, isPad: %ld, "
            "tailBlockSize: %ld, tilingKey: %lu, quantAxisSize: %ld, postAxisSize: %ld, nAlignSize: %ld "
            "mAlignBlockCount: %ld, nAlignBlockCount: %ld, mAlignGroupCount: %ld, "
            "quantAxisIsOdd: %ld, totalGroupNum: %ld, groupPerUb: %ld, "
            "totalBlockNum: %ld, blockNumPerTask: %ld, totalTaskNum: %ld, loopNumPerHeadCore: %ld, "
            "loopNumPerTailCore: %ld, ubRowLen: %ld, ubRowLenTail: %ld, ubRowCount: %ld, "
            "ubRowCountTail: %ld, subNumForScale: %ld, blockCountPerBatch: %ld,scaleRowCountPerBatch: %ld, "
            "needPadPostAxis: %ld, invDstTypeMax:%f, maxLowBound:%f.",
            tilingData_->totalCoreNum, tilingData_->usedCoreNum, tilingData_->blockSize, tilingData_->isPad,
            tilingData_->tailBlockSize, tilingData_->tilingKey, tilingData_->quantAxisSize, tilingData_->postAxisSize,
            tilingData_->nAlignSize, tilingData_->mAlignBlockCount, tilingData_->nAlignBlockCount,
            tilingData_->mAlignGroupCount, tilingData_->quantAxisIsOdd, tilingData_->totalGroupNum,
            tilingData_->groupPerUb, tilingData_->totalBlockNum, tilingData_->blockNumPerTask,
            tilingData_->totalTaskNum, tilingData_->loopNumPerHeadCore, tilingData_->loopNumPerTailCore,
            tilingData_->ubRowLen, tilingData_->ubRowLenTail, tilingData_->ubRowCount, tilingData_->ubRowCountTail,
            tilingData_->subNumForScale, tilingData_->blockCountPerBatch, tilingData_->scaleRowCountPerBatch,
            tilingData_->needPadPostAxis, tilingData_->invDstTypeMax, tilingData_->maxLowBound);
}

} // namespace optiling
