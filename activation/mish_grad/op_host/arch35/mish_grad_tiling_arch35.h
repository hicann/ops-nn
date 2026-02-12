/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MISHGRAD_TILING_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MISHGRAD_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "atvoss/elewise/elewise_tiling.h"

namespace optiling {
    struct MishGradCompileInfo {
        uint64_t coreNum = 0;
        uint64_t ubSize = 0;
    };

    class MishGradTiling {
    public:
        explicit MishGradTiling(gert::TilingContext* context) : tilingContext(context) {};
        ge::graphStatus RunTiling();
    protected:
        ge::graphStatus CalcOutputDtype();
        ge::graphStatus CalcInputDtype();
        ge::graphStatus CheckShape();

    private:
        gert::TilingContext* tilingContext;
        ge::DataType outputDtype;
        ge::DataType inputDtype;
        ge::DataType inputDtype1;
        ge::DataType inputDtype2;
        bool unfullCompute = true;
        uint64_t dType = 0;
    };
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_MISHGRAD_ARCH35_TILING_H