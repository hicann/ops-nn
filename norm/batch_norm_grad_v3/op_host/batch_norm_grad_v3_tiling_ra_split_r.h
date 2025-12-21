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
 * \file batch_norm_grad_v3_tiling_ra_split_r.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_TILING_RA_SPLIT_H__
#define __BATCH_NORM_GRAD_V3_TILING_RA_SPLIT_H__

#include "batch_norm_grad_v3_tiling.h"

namespace optiling {

class BatchNormGradV3TilingRASplitR : public BatchNormGradV3TilingBase {
public:
    explicit BatchNormGradV3TilingRASplitR(gert::TilingContext* context)
        : BatchNormGradV3TilingBase(context, baseTilingData_)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void PrintTilingData();
    ge::graphStatus Stage0Stage1UbTiling();
    ge::graphStatus Stage2UbTiling();

private:
    int64_t usedCoreNum_{0};
    int64_t rLoopFactor_{0};
    int64_t blockFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t binaryBlockCnt_{0};
    int64_t binaryFoldPoint_{0};
    int64_t binaryBlockTail_{0};
    int64_t lastCoreBlockCnt_{0};
    int64_t lastCoreFoldPoint_{0};
    int64_t lastCoreLoopTail_{0};
    int64_t aFactor_{0};
    int64_t aFactorAlign_{0};
    int64_t aFactorTail_{0};
    int64_t aLoopTimes_{0};
    int64_t dxLoopFactor_{0};
    int64_t dxLoopTail_{0};
    int64_t dxLoopTimes_{0};
    int64_t dxLastCoreFactor_{0};
    int64_t dxLastCoreTail_{0};
    int64_t dxLastCoreTimes_{0};
    int64_t cacheBuffCnt_{0};
    int64_t dyTypeSize_{0};
    BatchNormGradV3BaseTilingData baseTilingData_;
    BatchNormGradV3RASplitRTilingData tilingData_;
};

} // namespace optiling

#endif // __BATCH_NORM_GRAD_V3_TILING_RA_SPLIT_H__
