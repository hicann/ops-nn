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
 * \file segment_sum_simt_tiling.h
 * \brief segment_sum_simt_tiling
 */

#ifndef SEGMENT_SUM_SIMT_TILING_H
#define SEGMENT_SUM_SIMT_TILING_H

#include "segment_sum_tiling_base.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {

class SegmentSumSimtTiling : public SegmentSumBaseTiling
{
public:
    explicit SegmentSumSimtTiling(gert::TilingContext* context) : SegmentSumBaseTiling(context)
    {}
    ~SegmentSumSimtTiling() override
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
    SegmentSumSimtTilingData* tilingData_;
    uint64_t initNumPerCore_{0};
    uint64_t initNumTailCore_{0};
    uint32_t isDeterministic_{0};
    uint32_t maxSegIdsInUb{0};
    uint64_t segIdsPerCore_{0};
    uint64_t segIdsTailCore_{0};
    uint32_t segIdsPerLoop_{0};
    uint32_t segIdsPerLoopTailCore_{0};
    uint32_t segIdsTailLoop_{0};
    uint32_t segIdsTailLoopTailCore_{0};
    uint32_t loopTimes_{0};
    uint32_t loopTimesTailCore_{0};
};
} // namespace optiling
#endif // SEGMENT_SUM_SIMT_TILING_H