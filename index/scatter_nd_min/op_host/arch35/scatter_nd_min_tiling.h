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
 * \file scatter_nd_min_tiling.h
 * \brief scatter_nd_min_tiling
 */

#ifndef SCATTER_ND_MIN_TILING_H
#define SCATTER_ND_MIN_TILING_H

#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_base_tiling.h"
#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_simd_sort_tiling.h"

#include "index/scatter_nd_common/op_host/arch35/scatter_nd_common_simt_tiling.h"

namespace optiling {

// // ---------------------------ScatterNdMin Simt Tiling---------------------------
class ScatterNdMinSimtTiling : public ScatterNdCommonSimtTiling
{
public:
    explicit ScatterNdMinSimtTiling(gert::TilingContext* context) : ScatterNdCommonSimtTiling(context)
    {}
    ~ScatterNdMinSimtTiling() override
    {}
};

// // ---------------------------ScatterNdMin Simt Sort Tiling---------------------------
// class ScatterNdMinSimtSortTiling : public ScatterNdCommonSimtSortTiling
// {
// public:
//     explicit ScatterNdMinSimtSortTiling(gert::TilingContext* context) : ScatterNdCommonSimtSortTiling(context)
//     {}
//     ~ScatterNdMinSimtSortTiling() override
//     {}
// };

// ---------------------------ScatterNdMin Simd Sort Tiling---------------------------
class ScatterNdMinSimdSortTiling : public ScatterNdCommonSimdSortTiling
{
public:
    explicit ScatterNdMinSimdSortTiling(gert::TilingContext* context) : ScatterNdCommonSimdSortTiling(context)
    {}
    ~ScatterNdMinSimdSortTiling() override
    {}
};


} // namespace optiling
#endif // SCATTER_ND_MIN_TILING_H
