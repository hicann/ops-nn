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
 * \file bucketize_v2_full_load_tiling
 * \brief 
 */

#ifndef CANN_BUCKETIZE_V2_FULL_LOAD_TILING_H
#define CANN_BUCKETIZE_V2_FULL_LOAD_TILING_H

#include "bucketize_v2_tiling.h"

namespace optiling {

class BucketizeV2FullLoadTiling : public BucketizeV2BaseTiling {
public:
    explicit BucketizeV2FullLoadTiling(gert::TilingContext* context) : BucketizeV2BaseTiling(context)
    {}
    ~BucketizeV2FullLoadTiling() override
    {}

private:
    void DoUBTiling();
    void DoBlockTiling();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    int64_t blockFactor_ {0};
    int64_t blockTail_ {0};
    int64_t ubFactor_ {0};
    int64_t usedCoreNum_ {0};
    int64_t boundBufSize_ {0};
    int64_t inUbSize_ {0};
    int64_t outUbSize_ {0};
    int64_t maxIter_ {0};
    int64_t ubBlockSize_ {0};
    int64_t vRegLength_ {0};
};

} // namespace optiling
#endif // CANN_BUCKETIZE_V2_FULL_LOAD_TILING_H