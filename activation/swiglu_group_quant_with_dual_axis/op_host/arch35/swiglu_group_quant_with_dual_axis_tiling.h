/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_with_dual_axis_tiling.h
 * \brief Tiling data definition and tiling entry declarations for SwigluGroupQuantWithDualAxis.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_H
#define OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_H

#include <cstdint>
#include "../../op_kernel/arch35/swiglu_group_quant_flags.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "log/log.h"

namespace optiling {

// Compile-time platform data, used when the runtime platform info is absent.
struct SwigluGroupQuantWithDualAxisCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(SwigluGroupQuantWithDualAxisTilingData)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, inputType);
TILING_DATA_FIELD_DEF(uint32_t, weightType);
TILING_DATA_FIELD_DEF(uint32_t, flags);
TILING_DATA_FIELD_DEF(int64_t, t);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, keep);
TILING_DATA_FIELD_DEF(int64_t, groupCount);
TILING_DATA_FIELD_DEF(int64_t, batchRows);
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(float, bias);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(int64_t, scale1RowBytes);
TILING_DATA_FIELD_DEF(int64_t, scale2PairRows);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(int64_t, usedCoreCount);
TILING_DATA_FIELD_DEF(int64_t, tileRows);
TILING_DATA_FIELD_DEF(int64_t, tileCols);
TILING_DATA_FIELD_DEF(int64_t, groupChunkSize);
TILING_DATA_FIELD_DEF(uint64_t, ubUserBytes);
TILING_DATA_FIELD_DEF(uint64_t, ubPlannedBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuantWithDualAxis, SwigluGroupQuantWithDualAxisTilingData)

class SwigluGroupQuantWithDualAxisTiling {
public:
    // Borrows the framework context only for the synchronous Tiling entry call.
    // Instances must not escape that call or outlive the context.
    explicit SwigluGroupQuantWithDualAxisTiling(gert::TilingContext* context) : context_(context) {}
    ~SwigluGroupQuantWithDualAxisTiling() = default;
    SwigluGroupQuantWithDualAxisTiling(const SwigluGroupQuantWithDualAxisTiling&) = delete;
    SwigluGroupQuantWithDualAxisTiling& operator=(const SwigluGroupQuantWithDualAxisTiling&) = delete;

    ge::graphStatus DoOpTiling();

private:
    static ge::graphStatus GetPlatformInfoCommon(gert::TilingContext* context, uint64_t& coreNum, uint64_t& ubSize);
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    void SetTilingData();
    ge::graphStatus SetTilingKey();

    ge::graphStatus GetClampLimitAttr(const gert::RuntimeAttrs* attrs);
    ge::graphStatus CheckWeightInfo();
    ge::graphStatus CheckGroupIndexInfo();
    ge::graphStatus CheckOutputInfo(ge::DataType xDtype, const gert::Shape& xStorageShape);
    void CalcCoreTiling();
    void CalcBlockTiling();

    gert::TilingContext* context_ = nullptr;
    uint64_t tilingKey_ = 0;
    SwigluGroupQuantWithDualAxisTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    ge::DataType dstType_ = ge::DT_FLOAT8_E4M3FN;
    ge::DataType weightDtype_ = ge::DT_FLOAT;
    int64_t quantMode_ = 0;
    int64_t bs_ = 0;
    int64_t batchRows_ = 0;
    int64_t d_ = 0;
    int64_t splitD_ = 0;
    int64_t g_ = 0;
    int64_t dFactor_ = 0;
    int64_t dLoop_ = 0;
    int64_t tailDFactor_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t plannedUbBytes_ = 0;
    float alpha_ = 1.0f;
    float bias_ = 0.0f;
    double clampLimit_ = 0.0;
    int64_t hasClampLimit_ = 0;
    int64_t outputOrigin_ = 0;
    bool hasWeight_ = false;
    bool hasGroupIndex_ = false;
};
} // namespace optiling

#endif // OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_H
