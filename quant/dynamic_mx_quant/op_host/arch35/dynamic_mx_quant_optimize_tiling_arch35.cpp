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

namespace optiling {
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr float NUM_SIX_FLOAT = 6.0;
constexpr float NUM_SEVEN_FLOAT = 7.0;
constexpr int64_t NUM_TEN = 10;
constexpr int64_t MODE_ZERO = 0;
constexpr int64_t MODE_ONE = 1;
constexpr int64_t MODE_TWO = 2;
constexpr int64_t MODE_THREE = 3;
constexpr int64_t DIGIT_TEN_THOUSAND = 10000;
constexpr int64_t N_ALIGN128 = 128;
constexpr int64_t BLOCK_PER_GROUP = 2;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;

ge::graphStatus DynamicMxQuantOptimzieTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicMxQuantOptimzieTiling DoTiling.");

    OP_CHECK_IF(
        SetTilingParam() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DynamicMxQuantOptimzieTiling SetTilingParam Failed"),
        return ge::GRAPH_FAILED);
    // 设置计算模式，以此区分tilingkey与scale计算方式
    OP_CHECK_IF(
        SetCalcMode() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Set calculation mode Failed"),
        return ge::GRAPH_FAILED);
    SetTilingKey();
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

    OP_CHECK_IF(
        tilingParam_.groupPerUb == 0, OP_LOGE(context_->GetNodeName(), "Shape too large to fit the optimal template."),
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
    if (tilingParam_.postAxisSize <= N_ALIGN128) { // 小尾轴
        tilingParam_.totalGroupNum =
            tilingParam_.preAxisSize * tilingParam_.mAlignGroupCount * tilingParam_.nAlignBlockCount;
        tilingParam_.groupPerCore = Ops::Base::CeilDiv(tilingParam_.totalGroupNum, tilingParam_.totalCoreNum);
        tilingParam_.groupPerTail =
            (tilingParam_.groupPerCore == 0) ? 0 : tilingParam_.totalGroupNum % tilingParam_.groupPerCore;
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
        tilingParam_.ubRowLen = tilingParam_.vfLen / BYTES_OF_INPUT_TYPE;
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
        tilingParam_.scaleRowCountPerBatch =
            Ops::Base::CeilDiv(tilingParam_.quantAxisSize, NUM_TWO * tilingParam_.blockSize) * NUM_TWO;
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
        if (tilingParam_.dstTypeMax == NUM_ZERO || tilingParam_.dstTypeMax == NUM_SIX_FLOAT ||
            tilingParam_.dstTypeMax == NUM_SEVEN_FLOAT) {
            tilingParam_.calcMode = MODE_TWO;
        } else {
            tilingParam_.calcMode = MODE_THREE;
        }
    } else {
        OP_LOGE(context_->GetNodeName(), "Invalid scaleAlg value: %ld", tilingParam_.scaleAlg);
        return ge::GRAPH_FAILED;
    }

    if (tilingParam_.calcMode == MODE_TWO) {
        if (tilingParam_.dstTypeMax == NUM_ZERO || tilingParam_.dstTypeMax == NUM_SIX_FLOAT) {
            tilingParam_.subNumForScale = 0x000000c1;
        } else {
            tilingParam_.subNumForScale = 0x000000e1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantOptimzieTiling::SetTilingKey()
{
    // 万位数为1、2，分别代表融合尾轴大于128和融合尾轴小于等于128
    int64_t tenThousandDigit = tilingParam_.postAxisSize <= N_ALIGN128 ? NUM_ONE : NUM_TWO;
    // 十位数表示舍入模式，0,1,4分别代表round，floor，rint
    int64_t tenDigit = tilingParam_.roundMode;
    // 个位数表示计算模式，可取0,1,2,3
    int64_t digit = tilingParam_.calcMode;
    tilingParam_.tilingKey = tenThousandDigit * DIGIT_TEN_THOUSAND + tenDigit * NUM_TEN + digit;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::SetTilingData()
{
    tilingData.totalCoreNum = tilingParam_.totalCoreNum;
    tilingData.usedCoreNum = tilingParam_.usedCoreNum;
    tilingData.blockSize = tilingParam_.blockSize;
    tilingData.isPad = tilingParam_.isPad ? 1 : 0;
    tilingData.tailBlockSize = tilingParam_.tailBlockSize;
    tilingData.tilingKey = tilingParam_.tilingKey;
    tilingData.quantAxisSize = tilingParam_.quantAxisSize;
    tilingData.postAxisSize = tilingParam_.postAxisSize;
    tilingData.nAlignSize = tilingParam_.nAlignSize;
    tilingData.mAlignBlockCount = tilingParam_.mAlignBlockCount;
    tilingData.nAlignBlockCount = tilingParam_.nAlignBlockCount;
    tilingData.mAlignGroupCount = tilingParam_.mAlignGroupCount;
    tilingData.quantAxisIsOdd = tilingParam_.quantAxisIsOdd;
    tilingData.totalGroupNum = tilingParam_.totalGroupNum;
    tilingData.groupPerUb = tilingParam_.groupPerUb;
    tilingData.totalBlockNum = tilingParam_.totalBlockNum;
    tilingData.blockNumPerTask = tilingParam_.blockNumPerTask;
    tilingData.totalTaskNum = tilingParam_.totalTaskNum;
    tilingData.loopNumPerHeadCore = tilingParam_.loopNumPerHeadCore;
    tilingData.loopNumPerTailCore = tilingParam_.loopNumPerTailCore;
    tilingData.ubRowLen = tilingParam_.ubRowLen;
    tilingData.ubRowLenTail = tilingParam_.ubRowLenTail;
    tilingData.ubRowCount = tilingParam_.ubRowCount;
    tilingData.ubRowCountTail = tilingParam_.ubRowCountTail;
    tilingData.subNumForScale = tilingParam_.subNumForScale;
    tilingData.blockCountPerBatch = tilingParam_.blockCountPerBatch;
    tilingData.scaleRowCountPerBatch = tilingParam_.scaleRowCountPerBatch;
    tilingData.needPadPostAxis = tilingParam_.needPadPostAxis ? 1 : 0;
    tilingData.invDstTypeMax = tilingParam_.invDstTypeMax;

    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(
        rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantOptimzieTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "TilingData totalCoreNum: %ld, usedCoreNum: %ld, blockSize: %ld, isPad: %ld, "
        "tailBlockSize: %ld, tilingKey: %ld, quantAxisSize: %ld, postAxisSize: %ld, nAlignSize: %ld "
        "mAlignBlockCount: %ld, nAlignBlockCount: %ld, mAlignGroupCount: %ld, "
        "quantAxisIsOdd: %ld, totalGroupNum: %ld, groupPerUb: %ld, "
        "totalBlockNum: %ld, blockNumPerTask: %ld, totalTaskNum: %ld, loopNumPerHeadCore: %ld, "
        "loopNumPerTailCore: %ld, ubRowLen: %ld, ubRowLenTail: %ld, ubRowCount: %ld, "
        "ubRowCountTail: %ld, subNumForScale: %ld, blockCountPerBatch: %ld,scaleRowCountPerBatch: %ld, "
        "needPadPostAxis: %ld, invDstTypeMax:%f.",
        tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.blockSize, tilingData.isPad,
        tilingData.tailBlockSize, tilingData.tilingKey, tilingData.quantAxisSize, tilingData.postAxisSize,
        tilingData.nAlignSize, tilingData.mAlignBlockCount, tilingData.nAlignBlockCount, tilingData.mAlignGroupCount,
        tilingData.quantAxisIsOdd, tilingData.totalGroupNum, tilingData.groupPerUb, tilingData.totalBlockNum,
        tilingData.blockNumPerTask, tilingData.totalTaskNum, tilingData.loopNumPerHeadCore,
        tilingData.loopNumPerTailCore, tilingData.ubRowLen, tilingData.ubRowLenTail, tilingData.ubRowCount,
        tilingData.ubRowCountTail, tilingData.subNumForScale, tilingData.blockCountPerBatch,
        tilingData.scaleRowCountPerBatch, tilingData.needPadPostAxis, tilingData.invDstTypeMax);
}

} // namespace optiling
