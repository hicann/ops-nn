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
 * \file unsorted_segment_sum_tiling.h
 * \brief unsorted_segment_sum_tiling
 */

#ifndef UNSORTED_SEGMENT_SUM_TILING_H
#define UNSORTED_SEGMENT_SUM_TILING_H

#include <cstdint>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/tiling_context.h"

using namespace Ops::NN::Optiling;

namespace optiling {

BEGIN_TILING_DATA_DEF(UnsortedSegmentSumTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputOuterDim); // totalSampleNum_
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum, UnsortedSegmentSumTilingData);

ge::graphStatus Tiling4UnsortedSegmentSumForAscendC(gert::TilingContext* context);

class UnsortedSegmentSumBaseTiling : public TilingBaseClass
{
public:
    explicit UnsortedSegmentSumBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~UnsortedSegmentSumBaseTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override
    {}
    ge::graphStatus CheckInputDtype();
    bool ShapeStartsWith(const gert::Shape shape, const gert::Shape prefix);
    std::tuple<int64_t, int64_t> FlatInput(const gert::Shape shape, const gert::Shape prefix);
    uint32_t GetSortTmpSize(uint32_t lastAxisNum, bool isDescend);
    std::set<uint64_t> FindUniqueCut(uint64_t usedCoreNum);
    std::tuple<uint64_t, uint64_t> AutoTiling(
        uint64_t usedCoreNum, uint64_t colNumAlign, uint64_t colLimitSize, bool colTileNumMin = false);

public:
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t ubBlockSize_ = 0;
    uint32_t maxThread_ = 1;

    uint64_t inputOuterDim_ = 1;
    uint64_t outputOuterDim_ = 1;
    uint64_t innerDim_ = 1;
    uint64_t innerDimAlign_ = 1;
    uint64_t valueTypeBytes_ = 0;
    uint64_t idTypeBytes_ = 0;
    uint64_t dataShapeSize_ = 0;
    uint64_t ratio_ = 0;
    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType idType_ = ge::DT_UNDEFINED;

    uint64_t usrWorkspaceSize_ = 0;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_TILING_H
