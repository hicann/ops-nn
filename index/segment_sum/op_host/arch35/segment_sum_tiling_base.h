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
 * \file segment_sum_base_tiling.h
 * \brief segment_sum_base_tiling
 */

#ifndef SEGMENT_SUM_TILING_H
#define SEGMENT_SUM_TILING_H

#include <cstdint>

 	 
#include "kernel_tiling/kernel_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/atvoss/broadcast/broadcast_tiling.h"
#include "index/segment_sum/op_kernel/arch35/segment_sum_struct.h"

namespace optiling {

struct SegmentSumCompileInfo {
  uint64_t core_num;
  uint64_t ub_size;
};

class SegmentSumBaseTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit SegmentSumBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~SegmentSumBaseTiling() override
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
    virtual void SetTilingData() = 0;
    ge::graphStatus CheckInputDtype();

public:
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    // uint64_t usedCoreNum_ = 0;
    uint64_t ubBlockSize_ = 0;

    uint64_t outerDim_ = 1;
    uint64_t innerDim_ = 1;
    int64_t segmentNum_ = 0;
    uint64_t valueTypeBytes_ = 0;
    uint64_t idTypeBytes_ = 0;
    uint64_t dataShapeSize_ = 0;
    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType idType_ = ge::DT_UNDEFINED;

    uint64_t usrWorkspaceSize_ = 0;
};
} // namespace optiling
#endif // SEGMENT_SUM_TILING_H
