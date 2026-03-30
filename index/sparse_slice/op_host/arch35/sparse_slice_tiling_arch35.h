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
 * \file sparse_slice_tiling.h
 * \brief
 */

#ifndef OP_NN_INDEX_SPARSE_SLICE_OP_HOST_ARCH35_TILING_H
#define OP_NN_INDEX_SPARSE_SLICE_OP_HOST_ARCH35_TILING_H

#include <cstdint>
#include <vector>
#include "op_api/runtime2_util.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling {
using namespace Ops::NN::Optiling;
constexpr int64_t RANK_DIM_LIMIT = 24;

BEGIN_TILING_DATA_DEF(SparseSliceTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, valueNumbers);
TILING_DATA_FIELD_DEF(int64_t, rankNumbers);
TILING_DATA_FIELD_DEF(int64_t, valuePerUb);
TILING_DATA_FIELD_DEF(int64_t, valuePerCore);
TILING_DATA_FIELD_DEF(int64_t, valuePerTail);
TILING_DATA_FIELD_DEF(int64_t, placeHolderOne);
TILING_DATA_FIELD_DEF(int64_t, placeHolderTwo);
TILING_DATA_FIELD_DEF_ARR(int64_t, RANK_DIM_LIMIT, yShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, RANK_DIM_LIMIT, sliceStart);
TILING_DATA_FIELD_DEF_ARR(int64_t, RANK_DIM_LIMIT, sliceEnd);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseSlice, SparseSliceTilingData)

struct SparseSliceCompileInfo {
    int32_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct SparseSliceTilingParam {
    int64_t totalCoreNum{0}; // 总共有多少个核
    int64_t ubSize{0}; // 普通模板UB里面至少要放一个value，该value对应的indices，输入的shape，start的indices，size的
    uint32_t vfLen{0};
    int64_t tilingKey{0};
    int64_t workspaceSize{0};
    int64_t templateType{1}; // 模板类型
    int64_t usedCoreNum{0};  // 当前计算实际使用核数
    int64_t valueNumbers{0}; // 总共输入的values的数量
    int64_t rankNumbers{0};  // 输入shape的维度数量
    int64_t valuePerUb{0};   // 每块UB最大计算的value数量
    int64_t valuePerCore{0}; // 每个普通核计算的value数量
    int64_t valuePerTail{0}; // 每个尾核计算的value数量
    bool IsEmptyYShape{false};
    gert::Shape shape;
    gert::Shape start;
    gert::Shape size;
    int64_t yShapeOut[RANK_DIM_LIMIT]{0};
    int64_t sliceStart[RANK_DIM_LIMIT]{0};
    int64_t sliceEnd[RANK_DIM_LIMIT]{0};
};

class SparseSliceTiling : public TilingBaseClass {
public:
    explicit SparseSliceTiling(gert::TilingContext* context_) : TilingBaseClass(context_)
    {}
    ~SparseSliceTiling() override
    {}

    const string nodeName = "SparseSlice";
    SparseSliceTilingData tilingData;
    SparseSliceTilingParam tilingParams;

    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void PrintTilingData();
    ge::graphStatus CalcYShape();
    void GetValueList(size_t idx, const gert::Tensor* tensor, int64_t size, gert::Shape& valueList);

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool UseSIMT();
};
} // namespace optiling
#endif // OP_NN_INDEX_SPARSE_SLICE_OP_HOST_ARCH35_TILING_H