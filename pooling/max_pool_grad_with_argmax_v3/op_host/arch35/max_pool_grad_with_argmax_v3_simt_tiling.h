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

BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxV3SimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, nDim);
TILING_DATA_FIELD_DEF(int64_t, cDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kSizeH);
TILING_DATA_FIELD_DEF(int64_t, kSizeW);
TILING_DATA_FIELD_DEF(int64_t, stridesH);
TILING_DATA_FIELD_DEF(int64_t, stridesW);
TILING_DATA_FIELD_DEF(int64_t, padH);
TILING_DATA_FIELD_DEF(int64_t, padW);
TILING_DATA_FIELD_DEF(int64_t, dilationH);
TILING_DATA_FIELD_DEF(int64_t, dilationW);
TILING_DATA_FIELD_DEF(int64_t, ceilMode);
END_TILING_DATA_DEF;


REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_900, MaxPoolGradWithArgmaxV3SimtTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_901, MaxPoolGradWithArgmaxV3SimtTilingData);

class MaxPoolGradWithArgmaxV3SimtTiling : public MaxPoolGradWithArgmaxV3BaseTiling
{
public:
    explicit MaxPoolGradWithArgmaxV3SimtTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxV3BaseTiling(context)
    {
    }

    ~MaxPoolGradWithArgmaxV3SimtTiling() override
    {
    }

private:
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void PrintTilingData();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t tilingKey_ = 0;
    MaxPoolGradWithArgmaxV3SimtTilingData tilingData_;
};

}  // namespace optiling

#endif