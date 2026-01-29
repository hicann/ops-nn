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
 * \file max_pool_grad_with_argmax_v3_ksize_one_tiling.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_V3_KSIZE_ONE_TILING_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_V3_KSIZE_ONE_TILING_H_

#include "max_pool_grad_with_argmax_v3_tiling_base.h"

namespace optiling
{

BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxV3KSizeOneTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, coreLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, tailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;


REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_800, MaxPoolGradWithArgmaxV3KSizeOneTilingData);

struct MaxPoolGradWithArgmaxV3NHWCBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t moveDataNumCacheLineT2{0};
};

class MaxPoolGradWithArgmaxV3KsizeOneTiling : public MaxPoolGradWithArgmaxV3BaseTiling
{
public:
    explicit MaxPoolGradWithArgmaxV3KsizeOneTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxV3BaseTiling(context)
    {
    }

    ~MaxPoolGradWithArgmaxV3KsizeOneTiling() override
    {
    }

private:
    void DoUBTiling();
    void InitializationVars();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintTilingData() const;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t coreLoop_ = 0;
    int64_t tailCoreLoop_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t tailCoreTailUbFactor_ = 0;
    int64_t tilingKey_ = 0;

    MaxPoolGradWithArgmaxV3KSizeOneTilingData tilingData_;
    MaxPoolGradWithArgmaxV3NHWCBaseInfo baseData_;
};

}  // namespace optiling

#endif