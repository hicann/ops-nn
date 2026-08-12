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
 * \file bucketize_v2_simt_tiling.h
 * \brief
 */

#ifndef CANN_BUCKETIZE_V2_SIMT_TILING_H
#define CANN_BUCKETIZE_V2_SIMT_TILING_H

#include "bucketize_v2_tiling.h"

namespace optiling {

class BucketizeV2SimtTiling : public BucketizeV2BaseTiling {
public:
    explicit BucketizeV2SimtTiling(gert::TilingContext* context) : BucketizeV2BaseTiling(context) {}
    ~BucketizeV2SimtTiling() override {}

private:
    void DoBlockTiling();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    int64_t usedCoreNum_{0};
    int64_t maxIter_{0};
    bool IsUsedInt64_{true};
};

} // namespace optiling
#endif // CANN_BUCKETIZE_V2_SIMT_TILING_H
