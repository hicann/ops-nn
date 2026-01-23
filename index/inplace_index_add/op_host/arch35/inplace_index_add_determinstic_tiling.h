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
 * \file inplace_index_add_determinstic_tiling.h
 * \brief
 */

#ifndef INPLACE_INDEX_ADD_DETERMINSTIC_TILING_H_
#define INPLACE_INDEX_ADD_DETERMINSTIC_TILING_H_

#include "inplace_index_add_tiling_arch35.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(InplaceIndexAddDeterminsticTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);

/* for determinstic */
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumAfter);
TILING_DATA_FIELD_DEF(int64_t, ubIndexFactor);
TILING_DATA_FIELD_DEF(int64_t, afterAxisFactor);
TILING_DATA_FIELD_DEF(int64_t, ubVarOptiFactor);
/* pre */
TILING_DATA_FIELD_DEF(int64_t, eachCorePreAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailCorePreAxisCount);
TILING_DATA_FIELD_DEF(int64_t, updateLoopSize);
TILING_DATA_FIELD_DEF(int64_t, updateTailNum);
TILING_DATA_FIELD_DEF(int64_t, indicesLoopSize);
TILING_DATA_FIELD_DEF(int64_t, indiceAxisTailNum);
TILING_DATA_FIELD_DEF(int64_t, isSplitPreAxis);
/* after */
TILING_DATA_FIELD_DEF(int64_t, eachCoreAfterAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAfterAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailUpdateLoopSize);
TILING_DATA_FIELD_DEF(int64_t, tailUpdateAxisNum);
TILING_DATA_FIELD_DEF(int64_t, isSplitAfterAxis);

TILING_DATA_FIELD_DEF(int64_t, ubQuantaIndxFactor);
TILING_DATA_FIELD_DEF(int64_t, ubVarFactor);
TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, eachCoreVarCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreVarCount);
TILING_DATA_FIELD_DEF(int64_t, isWithAlpha);
TILING_DATA_FIELD_DEF(int64_t, isDeterminstic);
TILING_DATA_FIELD_DEF(int64_t, isOpti);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_300000, InplaceIndexAddDeterminsticTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_300001, InplaceIndexAddDeterminsticTilingData)
REGISTER_TILING_DATA_CLASS(InplaceIndexAdd_300002, InplaceIndexAddDeterminsticTilingData)

class InplaceIndexAddDeterminsticTiling : public InplaceIndexAddTiling
{
public:
    explicit InplaceIndexAddDeterminsticTiling(gert::TilingContext* context) : InplaceIndexAddTiling(context)
    {}
    ~InplaceIndexAddDeterminsticTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    int64_t GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes,
            int64_t originalSize, int64_t postAxisSize, ge::DataType idType);
    void DoOpTilingForDeterminsticSplitPre();
    void DoOpTilingForDeterminsticSplitAfter();
    void DoOpTilingForDeterminstic();
    void SetTilingData();
private:
    InplaceIndexAddDeterminsticTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_INPLACE_INDEX_ADD_TILING_H_