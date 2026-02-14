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
 * \file masked_scatter_with_position_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MASKED_SCATTER_WITH_POSITION_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MASKED_SCATTER_WITH_POSITION_H_

#include "register/op_def_registry.h"
#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/masked_scatter_with_position_tiling_struct.h"

namespace optiling {

class MaskedScatterWithPositionTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit MaskedScatterWithPositionTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckDataType();
    bool CanBroadcastBAOrAB(const gert::Shape xShape, const gert::Shape maskShape);
    void CanBroadcastBAOrABEqual(const std::vector<int64_t>& xShapeVec, const std::vector<int64_t>& maskShapeVec, const size_t len, bool& BA, bool& AB);

private:
    const char* opName_ = "MaskedScatterWithPosition";
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    int64_t xEleNums_ = 0;
    int64_t maskEleNums_ = 0;
    int64_t updatesEleNums_ = 0;
    ge::DataType dType_ = ge::DT_UNDEFINED;
    bool isInt64Mask_ = false;
    bool isBA_ = false;
    int64_t xOuter_ = 1;
    int64_t xInner_ = 1;

    int64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;
};

struct MaskedScatterWithPositionCompileInfo {
    int64_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // MASKED_SCATTER_WITH_POSITION_TILLING_DATA_H
