/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License")
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
* \file scatter_add_with_sorted_simt_tiling.h
* \brief scatter_add_with_sorted_simt_tiling
*/
#ifndef SCATTER_ADD_WITH_SORTED_SIMT_TILING_H
#define SCATTER_ADD_WITH_SORTED_SIMT_TILING_H

#include "scatter_add_with_sorted_tiling_base.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {

class ScatterAddWithSortedSimtTiling : public ScatterAddWithSortedBaseTiling
{
public:
    explicit ScatterAddWithSortedSimtTiling(gert::TilingContext* context) : ScatterAddWithSortedBaseTiling(context)
    {}
    ~ScatterAddWithSortedSimtTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData() override;

private:
    ScatterAddWithSortedSimtTilingData* tilingData_;
    int64_t normBlockIndices_ = 0;
    int64_t tailBlockIndices_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t workspacesSize_ = 0;
    
};
}
#endif // SCATTER_ADD_WITH_SORTED_SIMT_TILING_H