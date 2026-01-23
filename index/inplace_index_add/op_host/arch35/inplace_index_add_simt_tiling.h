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
 * \file inplace_index_add_simt_tiling.h
 * \brief
 */

#ifndef INPLACE_INDEX_ADD_SIMT_TILING_H_
#define INPLACE_INDEX_ADD_SIMT_TILING_H_

#include "inplace_index_add_tiling_arch35.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(InplaceIndexAddSimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, colUbFactor);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
TILING_DATA_FIELD_DEF(int64_t, indicesUbFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100000, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100001, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100002, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100003, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100004, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100006, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100027, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100009, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100012, InplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100100, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100101, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100102, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100103, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100104, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100106, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100127, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100109, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_100112, InplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101000, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101001, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101002, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101003, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101004, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101006, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101027, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101009, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101012, InplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101100, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101101, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101102, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101103, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101104, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101106, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101127, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101109, InplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_101112, InplaceIndexAddSimtTilingData)

class InplaceIndexAddSimtTiling : public InplaceIndexAddTiling
{
public:
    explicit InplaceIndexAddSimtTiling(gert::TilingContext* context) : InplaceIndexAddTiling(context)
    {}
    ~InplaceIndexAddSimtTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();
    InplaceIndexAddSimtTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_INPLACE_INDEX_ADD_TILING_H_