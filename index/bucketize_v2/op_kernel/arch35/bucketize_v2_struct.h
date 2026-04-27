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
 * \file bucketize_v2_struct.h
 * \brief tiling base data
 */

#ifndef BUCKETIZE_V2E_STRUCT_H
#define BUCKETIZE_V2E_STRUCT_H
class BucketizeV2TilingData {
    public:
        int64_t xSize;
        int64_t boundDtypeSize;
};
class BucketizeV2FullLoadTilingData {
    public:
        int64_t usedCoreNum = 0;
        int64_t blockFactor = 0;
        int64_t blockTail = 0;
        int64_t ubFactor = 0;
        int64_t boundBufSize = 0;
        int64_t inUbSize = 0;
        int64_t outUbSize = 0;
        int64_t boundSize = 0;
        int64_t maxIter = 0;
};

class BucketizeV2CascadeTilingData{
    public:
        int64_t usedCoreNum = 0;
        int64_t blockFactor = 0;
        int64_t blockTail = 0;
        int64_t ubFactor = 0;
        int64_t boundBufSize = 0;
        int64_t inUbSize = 0;
        int64_t outUbSize = 0;
        int64_t midOutUbSize = 0;
        int64_t boundSize = 0;
        int64_t sampleBoundSize = 0;
        int64_t sampleRatio = 0;
        int64_t outerMaxIter = 0;
        int64_t innerMaxIter = 0;
};
#endif