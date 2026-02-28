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
 * \file max_pool_with_argmax_v3_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_WITH_AGRMAX_V3_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_WITH_AGRMAX_V3_TILING_BASE_H_

#include <array>

#include <cstdint>
#include <vector>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_common/op_host/util/platform_util.h"

using namespace std;

namespace optiling {
const int HW_DIMS = 2;
const int HW_PAD_DIMS = 5;
const int MAX_CORE_NUM = 64;
const uint32_t H_DIM = 0;
const uint32_t W_DIM = 1;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;
const uint32_t BLOCK_LEN_FP32 = 8;
const uint32_t BLOCK_LEN_FP16 = 16;

BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxV3TilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kh);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3, MaxPoolWithArgmaxV3TilingData);

struct InputInfo {
    uint64_t batches;
    array<uint64_t, HW_DIMS> inputShape;
    array<uint64_t, HW_DIMS> outShape;
    array<uint64_t, HW_DIMS> kernelSize;
    array<uint64_t, HW_DIMS> stride;
    array<uint64_t, HW_DIMS> pad;
    array<uint64_t, HW_DIMS> dilation;
    bool ceilMode;
    ge::DataType indexDtype;
    ge::Format inputFormat;
    uint64_t nInput;
    uint64_t cInput;
};

struct MaxPoolWithArgmaxV3CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPoolWithArgmaxV3BaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MaxPoolWithArgmaxV3BaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    ~MaxPoolWithArgmaxV3BaseTiling() override
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

public:
    InputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint32_t coreNum = 1;
    uint32_t ubSize = 0;
};
} // namespace optiling
#endif