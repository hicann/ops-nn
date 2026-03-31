/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_tiling_base.h
 * \brief
 */
#ifndef INPLACE_INDEX_FILL_TILING_BASE_H_
#define INPLACE_INDEX_FILL_TILING_BASE_H_

#include "op_host/tiling_base.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
using namespace Ops::NN::Optiling;

const uint64_t INPUT_X_IDX = 0;
const uint64_t INPUT_INDICES_IDX = 1;
const uint64_t INPUT_VALUE_IDX = 2;
const uint64_t OUTPUT_X_IDX = 0;
const uint64_t ATTR_IDX = 0;
const uint32_t SYS_WORKSPACE_SIZE = 0;

struct InplaceIndexFIllInputInfo {
    int64_t dim;        // 指定的维度，用于填充操作
    int64_t preDimProduct = 1;      // x在dim轴前面的轴维度乘积
    int64_t dimSize = 0;            // dim轴的维度
    int64_t postDimProduct = 1;     // x在dim轴后面的轴维度乘积
    int64_t indicesNum; // indices的长度
    int64_t totalDataSize;   // 需处理的总数据量
    int64_t xDtypeSize;
    int64_t indicesDtypeSize;
};

struct InplaceIndexFillCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class InplaceIndexFillTilingBase : public TilingBaseClass
{
public:
    explicit InplaceIndexFillTilingBase(gert::TilingContext* context) : TilingBaseClass(context) 
    {}
    ~InplaceIndexFillTilingBase()
    {}

protected:
    InplaceIndexFIllInputInfo inputData;
    ge::graphStatus CheckDataType();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    void CalculatePQ(const gert::Shape& xShape, int64_t dim, int64_t xDim, InplaceIndexFIllInputInfo& inputDataParam);
    bool IsCapable();
    ge::graphStatus DoOpTiling();
    ge::graphStatus DoLibApiTiling();
    uint64_t GetTilingKey() const;

public:
    int64_t coreNum = 1;
    uint64_t ubSize = 0;
    int64_t usedCoreNum = 1;
};

ge::graphStatus Tiling4InplaceIndexFillArch35(gert::TilingContext* context);
}   // namespace optiling

#endif  // INPLACE_INDEX_FILL_TILING_BASE_H_