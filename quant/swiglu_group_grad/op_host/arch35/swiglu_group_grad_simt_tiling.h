/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_simt_tiling.h
 * \brief SwigluGroupGrad SIMT tiling class (arch35, Ascend950)
 */

#ifndef SWIGLU_GROUP_GRAD_SIMT_TILING_H_
#define SWIGLU_GROUP_GRAD_SIMT_TILING_H_

#include "swiglu_group_grad_tiling_base.h"

namespace optiling {

using Ops::NN::Optiling::TilingBaseClass;

class SwigluGroupGradSimtTiling : public TilingBaseClass {
public:
    explicit SwigluGroupGradSimtTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~SwigluGroupGradSimtTiling() override {}

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    SwigluGroupGradInputInfo inputData;
    SwigluGroupGradSimtTilingData* simtTiling = nullptr;
    int64_t coreNumAll_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t schMode_ = TPL_SIMT_KERNEL;
};

} // namespace optiling

#endif // SWIGLU_GROUP_GRAD_SIMT_TILING_H_
