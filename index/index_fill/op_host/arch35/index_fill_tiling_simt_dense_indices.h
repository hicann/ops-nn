/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_tiling_simt_dense_indices.h
 * \brief
 */

#ifndef INDEX_FILL_TILING_SIMT_DENSE_INDICES_H_
#define INDEX_FILL_TILING_SIMT_DENSE_INDICES_H_

#include "index_fill_tiling_simt.h"

namespace optiling
{
class IndexFillSimtDenseIndicesTiling : public IndexFillSimtTiling
{
public:
    explicit IndexFillSimtDenseIndicesTiling(gert::TilingContext* context) : IndexFillSimtTiling(context)
    {
    }
    ~IndexFillSimtDenseIndicesTiling() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void SetTilingData() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;

private:
    void DoIndicesTilingTask();
    int64_t GetOptimalIndiceUBFactor(int64_t indicesUbFactor, int64_t indicesNum, int64_t n);
    uint32_t GetSortTmpSize(ge::DataType indicesDtype, uint32_t shapeSize, bool isDescend);
    int64_t CalcUsedBufSize(int64_t indicesUbFactor, ge::DataType indicesDtype);
};

}  // namespace optiling
#endif  // INDEX_FILL_TILING_SIMT_DENSE_INDICES_H_
