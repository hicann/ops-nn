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
 * \file apply_adagrad_tiling_arch35.h
 * \brief ApplyAdagrad arch35 tiling declaration.
 */
#ifndef OPS_NN_APPLY_ADAGRAD_TILING_ARCH35_H
#define OPS_NN_APPLY_ADAGRAD_TILING_ARCH35_H

#include "op_host/tiling_base.h"
#include "../../op_kernel/apply_adagrad_struct.h"

using namespace ApplyAdagradTilingData;
namespace optiling {
struct ApplyAdagradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class ApplyAdagradTiling {
public:
    explicit ApplyAdagradTiling(gert::TilingContext* context) : tilingContext_(context) {};

    ge::graphStatus RunTiling();

protected:
    ge::graphStatus SetTilingData();
    bool CheckIsScalar(int32_t inputIdx);
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();
    ge::graphStatus ComputeTiling();
    ge::graphStatus InitTilingData();
    ge::graphStatus ComputeBlockTiling(int64_t coreNum);
    ge::graphStatus ComputeUbTiling(uint64_t ubSize);

private:
    ge::DataType varDtype_ = ge::DT_FLOAT;
    ApplyAdagradTilingDataStruct* tiling_ = nullptr;
    gert::TilingContext* tilingContext_;
    uint64_t updateSlots = 0;
    uint64_t dType = 0;
    int64_t totalElements_ = 0;
    int64_t blockNum_ = 1;
};
} // namespace optiling
#endif // OPS_NN_APPLY_ADAGRAD_TILING_ARCH35_H
