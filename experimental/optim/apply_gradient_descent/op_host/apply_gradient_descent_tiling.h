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
 * \file apply_gradient_descent_tiling.h
 * \brief apply_gradient_descent classic (ascend910b) tiling.
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_GRADIENT_DESCENT_TILING_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_GRADIENT_DESCENT_TILING_H_

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/apply_gradient_descent_tiling_data.h"

namespace optiling {

struct ApplyGradientDescentCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

class ApplyGradientDescentTiling {
public:
    explicit ApplyGradientDescentTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus RunTiling();

private:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    void SplitCore();
    void CalcTileDataCount();
    void SetTilingData(ApplyGradientDescentTilingData* tilingData);

    gert::TilingContext* context_ = nullptr;
    ge::DataType varDtype_ = ge::DT_FLOAT;
    uint32_t dtypeSize_ = 0;
    uint32_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t totalDataCount_ = 0;
    uint64_t tileDataCount_ = 0;
    uint64_t blocksPerCore_ = 0;
    uint32_t needCoreNum_ = 0;
    uint32_t blockElems_ = 0;
    uint32_t remCoreNum_ = 0;
    uint64_t tilingKey_ = 0;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_GRADIENT_DESCENT_TILING_H_
