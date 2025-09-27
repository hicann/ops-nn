/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file dynamic_quant_tiling_310P.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_QUANT_310P_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_QUANT_310P_TILING_H
#include <cstdint>
#include "dynamic_quant_tiling.h"

namespace optiling {

class DynamicQuantTiling310P
{
public:
    DynamicQuantTiling310P() = default;
    ~DynamicQuantTiling310P() = default;
    DynamicQuantTilingData tiling;
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

private:
    uint64_t ComputeTilingKey(const gert::TilingContext* context);
    ge::graphStatus SetTilingData(uint64_t rowNumTotal, gert::TilingContext* context);
    ge::graphStatus CorrectNumCopyRow(uint64_t rowNumTotal);
    void SetSuitNumCopyRow(const gert::TilingContext* context);
    ge::graphStatus ParseShape(const gert::TilingContext* context, uint64_t* rowNumTotal);
    ge::graphStatus CheckDtype(const gert::TilingContext* context) const;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_QUANT__TILING_H
