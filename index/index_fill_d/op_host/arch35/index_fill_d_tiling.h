/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_d_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_FILL_D_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_FILL_D_TILING_H
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {
using namespace Ops::NN::Optiling;

BEGIN_TILING_DATA_DEF(IndexFillDTilingData)
TILING_DATA_FIELD_DEF(int64_t, normalCoreData);
TILING_DATA_FIELD_DEF(int64_t, tailCoreData);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, tailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexFillD, IndexFillDTilingData)

struct IndexFillDCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class IndexFillDTiling : public TilingBaseClass
{
public:
    explicit IndexFillDTiling(gert::TilingContext* context) : TilingBaseClass(context) {};

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckShape();

private:
    ge::DataType dType_ = ge::DT_UNDEFINED;
    gert::Shape inputXShape_;
    int64_t inputShapeSize_ = 0;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t normalCoreData_ = 0;
    int64_t tailCoreData_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t tailCoreTailUbFactor_ =0;
    int64_t normalCoreLoop_ = 0;
    int64_t tailCoreLoop_ = 0;
    int64_t dataTypeSize_ = 1;
    IndexFillDTilingData tilingData_;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_FILL_D_TILING_H