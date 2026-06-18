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
 * \file max_pool3d_with_argmax_v2_ksize_one_tiling.h
 * \brief
 */

#ifndef MAX_POOL3D_WITH_ARGMAX_V2_KSIZE_ONE_TILING_H_
#define MAX_POOL3D_WITH_ARGMAX_V2_KSIZE_ONE_TILING_H_

#include "max_pool3d_with_argmax_v2_tiling_base.h"
#include "../op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h"

namespace optiling {

class MaxPool3DWithArgmaxV2KsizeOneTiling : public MaxPool3DWithArgmaxV2BaseTiling {
public:
    explicit MaxPool3DWithArgmaxV2KsizeOneTiling(gert::TilingContext* context)
        : MaxPool3DWithArgmaxV2BaseTiling(context)
    {}

    ~MaxPool3DWithArgmaxV2KsizeOneTiling() override {}

private:
    void InitializationVars();
    void DoTiling();
    void SetTilingData();
    void PrintBaseData() const;
    void PrintTilingData() const;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData* tilingData_ =
        context_->GetTilingData<MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData>();

    int64_t inputBytes_ = 0;
    int64_t indexBytes_ = 0;
    int64_t ncTotal_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t coreLoop_ = 0;
    int64_t tailCoreLoop_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t tailCoreTailUbFactor_ = 0;
    int64_t inputBufferSize_ = 0;
    int64_t argmaxBufferSize_ = 0;
};

} // namespace optiling

#endif