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
 * \file max_pool_v3_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_V3_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_V3_TILING_BASE_H_

#include <array>

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "common/op_util.h"

namespace optiling {
const int32_t HW_DIMS = 2;
const int32_t PAD_DIMS = 4;
const uint32_t H_DIM = 0;
const uint32_t W_DIM = 1;
const uint32_t TOP_PAD_INDEX = 0;
const uint32_t BOTTOM_PAD_INDEX = 1;
const uint32_t LEFT_PAD_INDEX = 2;
const uint32_t RIGHT_PAD_INDEX = 3;
const int32_t INPUT_KSIZE_IDX = 1;
const int32_t INPUT_STRIDES_IDX = 2;

BEGIN_TILING_DATA_DEF(MaxPoolV3TilingData)
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, tPad);
TILING_DATA_FIELD_DEF(int64_t, bPad);
TILING_DATA_FIELD_DEF(int64_t, lPad);
TILING_DATA_FIELD_DEF(int64_t, rPad);
TILING_DATA_FIELD_DEF(int64_t, channel);
TILING_DATA_FIELD_DEF(int64_t, coreNums);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, outUbSize);
TILING_DATA_FIELD_DEF(int64_t, tilingMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolV3, MaxPoolV3TilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolV2, MaxPoolV3TilingData);

struct InputInfo {
    int64_t batches;
    int64_t channels;
    std::array<int64_t, HW_DIMS> inputShape;
    std::array<int64_t, HW_DIMS> outShape;
    std::array<int64_t, HW_DIMS> kernelSize;
    std::array<int64_t, HW_DIMS> stride;
    std::array<int64_t, PAD_DIMS> pad;
    std::array<int64_t, HW_DIMS> dilation;
    bool ceilMode;
    bool globalPooling;
    ge::DataType indexDtype;
    ge::Format inputFormat;
};

struct MaxPoolV3CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPoolV3BaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MaxPoolV3BaseTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

    ~MaxPoolV3BaseTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetShapeAndDtype();
    ge::graphStatus GetKernelInfo();
    ge::graphStatus GetAttrsInfo();
    ge::graphStatus GetPadInfo();
    ge::graphStatus CheckOutPutShapeForValid();
    ge::graphStatus CheckOutPutShapeForSame();
    ge::graphStatus CheckOutPutShapeForCalculated();
    ge::graphStatus CheckOutPutShape();
    void RefineShape();
    ge::graphStatus CheckShape(gert::Shape& inputShape, gert::Shape& outputShape, const ge::Format& inputFormat);
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    bool IsInvalidType(const ge::DataType dataType);
    bool IsInvalidPaddingMode(std::string padMode);
    int64_t InferCalculateOutShape(
        const int64_t ksize, const int64_t padL, const int64_t padR, const int64_t stride, const bool ceilMode,
        int64_t dimSize);

public:
    InputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    int64_t dtypeSize = 0;
    int64_t coreNum = 1;
    uint32_t ubSize = 0;
    int32_t nDim_ = 0; // NCHW -> N
    int32_t cDim_ = 1; // NCHW -> C
    int32_t hDim_ = 2; // NCHW -> H
    int32_t wDim_ = 3; // NCHW -> W
    std::string padModeStr_;
};

} // namespace optiling

#endif