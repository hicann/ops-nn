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
 * \file repeat_interleave_tiling_base.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_BASE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_BASE_H_

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

using Ops::NN::Optiling::TilingBaseClass;
namespace optiling {
constexpr uint32_t REPEAT_INTERLEAVE_MERGED_DIM_LENGTH = 3;

class RepeatInterleaveBaseTiling : public TilingBaseClass {
public:
    explicit RepeatInterleaveBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~RepeatInterleaveBaseTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void MergDimForTensor();
    void MergDimForScalar();
    void MergDim();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();
    int64_t MergeDimExceptAxis(const gert::Shape& input, int64_t axis);
    inline bool IsSupportDtype(const std::set<ge::DataType>& supportDtype, const ge::DataType dtype)
    {
        return (supportDtype.count(dtype) != 0);
    }

protected:
    int64_t totalCoreNum_{0};
    int64_t ubSize_{0};
    int64_t usedCoreNum_{0};
    gert::Shape repeatShape_;
    gert::Shape inputShape_;
    gert::Shape yShape_;
    ge::DataType inputDtype_;
    ge::DataType repeatDtype_;
    int64_t mergedDim_[REPEAT_INTERLEAVE_MERGED_DIM_LENGTH] = {0};
    int64_t axis_{0};
    bool isDefaultAxis_{false};
    int64_t repeatsCount_{-1};
};
} // namespace optiling
#endif