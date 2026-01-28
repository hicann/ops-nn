/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file dynamic_quant_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_QUANT_REGBASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_QUANT_REGBASE_TILING_H
#include <cstdint>

#include "dynamic_quant_tiling.h"

namespace optiling {

class DynamicQuantRegbaseTiling
{
public:
    DynamicQuantRegbaseTiling() = default;
    ~DynamicQuantRegbaseTiling() = default;
    DynamicQuantTilingData tilingData;
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

private:
    void SetTilingKey(gert::TilingContext* context)const;
    ge::graphStatus CheckInputDtype(gert::TilingContext* context);
    ge::graphStatus CheckOutputDtype(gert::TilingContext* context);
    ge::graphStatus CheckOpInputShape(gert::TilingContext* context);
    ge::graphStatus CheckOpOutputShape(gert::TilingContext* context) const;
    ge::graphStatus CheckAttrs(gert::TilingContext* context);
    ge::graphStatus CheckOpShape(gert::TilingContext* context);
    ge::graphStatus CheckOpDim(
        const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim, uint32_t shape2Dim)const;
    ge::graphStatus CheckOpParams(gert::TilingContext* context);
    void ResetLargeTilingParams();
    void SetTilingData(gert::TilingContext* context);
    void PrintTilingData(gert::TilingContext* context);

    void IsCapableForFullLoad(gert::TilingContext* context);
    void IsCapableForRecompute(gert::TilingContext* context);
    void IsCapableForSplitM(gert::TilingContext* context);
    void SetTilingDataForPerChannel();
    void PrintTilingDataForPerChannel(gert::TilingContext* context);
    void CalculateTilingData();
    ge::graphStatus CalculateTilingDataForPerChannel(gert::TilingContext* context);
    void CalculateCoreNum(const gert::TilingContext* context);
    ge::graphStatus GetCompileInfo(gert::TilingContext* context);
    ge::graphStatus DoEmptyTensorTiling(gert::TilingContext* context);

private:
    uint32_t vectorCoreNum{0};
    uint64_t ubSize{0};

    uint32_t coreNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNumHeadCore;
    uint32_t multiRowNumTailCore;

    uint32_t rowNum;
    uint32_t rowNumPerMinTask;
    uint32_t scaleNumPerMinTask;
    uint32_t rowNumPerTask;
    uint32_t taskNum;
    uint32_t wholeTaskNum;
    uint32_t ubPerRow;
    uint32_t ubPerRowNew;

    uint32_t innerLoopEle = 0;
    uint32_t innerLoopTimes = 0;
    uint32_t innerLoopTail = 0;

    uint32_t groupNum = 0;
    uint32_t groupDtypeSize = 0;
    bool hasSmooth = false;
    bool useDb = false;
    bool isEmptyTensor = false;
    uint32_t quantMode_ = 0;     // 0: pertoken, 1: pertensor
    bool isSymmetrical_ = false; // 0: False, 1: True

    int32_t yDtype;
    // members for perchannel
    bool isPerChannel_ = false;
    int64_t mLen = 0;
    int64_t nLen = 0;
    int64_t totalBatchLen = 0;
    int64_t nBlockSize = 1;
    int64_t nBlockNum = 1;
    int64_t mBlockSize = 0;
    int64_t mBlockNum = 1;
    int64_t totalBlockNum = 0;
    int64_t blockPerHead = 0;
    int64_t blockPerTail = 0;
    int64_t nMaxLoopNum = 16;
    int64_t nBaseLoopNum = 1;

    int64_t batchBlockSize = 1;
    int64_t batchTailBlockSize = 1;
    int64_t batchBlockNum = 1;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_QUANT_REGBASE_TILING_H
