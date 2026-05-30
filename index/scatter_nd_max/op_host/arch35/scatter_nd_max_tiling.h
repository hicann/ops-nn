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
 * \file scatter_nd_max_tiling.h
 * \brief scatter_nd_max_tiling
 */

#ifndef SCATTER_ND_MAX_TILING_H
#define SCATTER_ND_MAX_TILING_H

#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_base_tiling.h"
#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_simd_sort_tiling.h"
#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_simt_sort_tiling.h"
#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_simt_tiling.h"

namespace optiling {

// ---------------------------ScatterNdMax Simt Tiling---------------------------
class ScatterNdMaxSimtTiling : public ScatterNdCommonSimtTiling
{
public:
    explicit ScatterNdMaxSimtTiling(gert::TilingContext* context) : ScatterNdCommonSimtTiling(context)
    {}
    ~ScatterNdMaxSimtTiling() override
    {}
};

// ---------------------------ScatterNdMax Simt Sort Tiling---------------------------
class ScatterNdMaxSimtSortTiling : public ScatterNdCommonSimtSortTiling
{
public:
    explicit ScatterNdMaxSimtSortTiling(gert::TilingContext* context) : ScatterNdCommonSimtSortTiling(context)
    {}
    ~ScatterNdMaxSimtSortTiling() override
    {}
};

// ---------------------------ScatterNdMax Simd Sort Tiling---------------------------
class ScatterNdMaxSimdSortTiling : public ScatterNdCommonSimdSortTiling
{
public:
    explicit ScatterNdMaxSimdSortTiling(gert::TilingContext* context) : ScatterNdCommonSimdSortTiling(context)
    {}
    ~ScatterNdMaxSimdSortTiling() override
    {}
};


} // namespace optiling
#endif // SCATTER_ND_MAX_TILING_H
