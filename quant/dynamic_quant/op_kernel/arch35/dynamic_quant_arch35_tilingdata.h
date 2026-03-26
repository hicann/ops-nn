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
 * \file dynamic_quant_tilingdata.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_TILINGDATA_H
#define DYNAMIC_QUANT_TILINGDATA_H

#include <cstdint>

class DynamicQuantTilingDataArch35 {
public:
    uint32_t coreNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNumHeadCore;
    uint32_t multiRowNumTailCore;
    uint32_t innerLoopEle;
    uint32_t innerLoopTimes;
    uint32_t innerLoopTail;
    uint32_t groupNum;
    uint32_t alignGroupNum;
    uint32_t hasSmooth;
    uint32_t unused;
    uint32_t ubSize;
    uint32_t sizeH;
    uint32_t sizeX;
    uint32_t sizeZOut;
    uint32_t sizeCopyRow;
    uint32_t numCopyRow;
    uint32_t numHeadCore;
    uint32_t numTailCore;
    uint32_t numHeadTimes;
    uint32_t numTailTimes;
    uint32_t numLastTailRow;
    uint32_t alignType;
    int64_t totalBatchLen;
    int64_t mLen;
    int64_t mBlockSize;
    int64_t mTailBlockSize;
    int64_t mBlockNum;
    int64_t nLen;
    int64_t nBlockSize;
    int64_t nTailBlockSize;
    int64_t nBlockNum;
    int64_t nBaseSize;
    int64_t nBaseLoopNum;
    int64_t blockPerHead;
    int64_t blockPerTail;
    int64_t totalBlockNum;
    int64_t batchBlockSize;
    int64_t batchTailBlockSize;
    int64_t batchBlockNum;
    float dstTypeMax;
};

#endif // DYNAMIC_QUANT_TILINGDATA_H
