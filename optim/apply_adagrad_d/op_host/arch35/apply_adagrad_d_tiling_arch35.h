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
 * \file apply_adagrad_d_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ADAGRAD_D_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ADAGRAD_D_TILING_H

#include "op_host/tiling_base.h"
#include "../../op_kernel/apply_adagrad_d_struct.h"

using namespace ApplyAdagradDTilingData;
namespace optiling {
struct ApplyAdagradDCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class ApplyAdagradDTiling
{
public:
    explicit ApplyAdagradDTiling(gert::TilingContext *context) : tilingContext_(context) {};

    ge::graphStatus RunTiling();

protected:
    ge::graphStatus SetTilingData();
    bool CheckIsScalar(int32_t inputIdx);
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();
    ge::graphStatus DoUpdateSlotsTiling();
    ge::graphStatus DoNonUpdateSlotsTiling();

private:
    ge::DataType varDtype_;
    ApplyAdagradDTilingDataStruct *tiling_;
    gert::TilingContext *tilingContext_;
    uint64_t schMode = 0;
    uint64_t updateSlots = 0;
    uint64_t dType = 0;
    bool updateSlots_;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ADAGRAD_D_TILING_H