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
* \file index_put_with_sort_v2_simd_tiling_arch35.h
* \brief IndexPutWithSortV2 tiling file
*/
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_WITH_SORT_V2_SIMD_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_WITH_SORT_V2_SIMD_ARCH35_H

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "index_put_with_sort_v2_tiling_arch35.h"

namespace optiling {

class IndexPutWithSortV2SIMDTiling : public IndexPutWithSortV2Tiling
{
public:
    explicit IndexPutWithSortV2SIMDTiling(gert::TilingContext* context) : IndexPutWithSortV2Tiling(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    int64_t inOutUb_ = 0;
    int64_t xCastDtypeSize_ = 0;
    bool isCast_ = false;

    int64_t indicesFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t rowBlockFactor_ = 1;
    int64_t rowUseCoreNum_ = 1;
    int64_t rowTailBlockFactor_ = 1;
    int64_t colBlockFactor_ = 1;
    int64_t colUseCoreNum_ = 1;
    int64_t colTailBlockFactor_ = 1;
    int64_t GetInputDtypeSize(ge::DataType dtype);
    void SetTilingData();
    void LogTilingResult();
    void DoBlockTiling();
    void DoUbTiling();
};
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_WITH_SORT_V2_H