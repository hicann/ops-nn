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
 * \file max_pool_with_argmax_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_WITH_AGRMAX_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_WITH_AGRMAX_TILING_BASE_H_

#include <array>
#include <cstdint>

#include "error_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling
{
using namespace std;

const int HW_DIMS = 2;
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

BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxTilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kh);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmax, MaxPoolWithArgmaxTilingData);

struct InputInfo {
    array<uint64_t, HW_DIMS> inputShape;
    array<uint64_t, HW_DIMS> outShape;
    array<uint64_t, HW_DIMS> kernelSize;
    array<uint64_t, HW_DIMS> stride;
    array<uint64_t, HW_DIMS> pad;
    uint64_t isPad;
    uint64_t includeBatchInIndex;
    uint64_t nanProp;
    ge::DataType indexDtype;
    ge::Format inputFormat;
    uint64_t nInput;
    uint64_t cInput;
};

struct MaxPoolWithArgmaxCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPoolWithArgmaxBaseTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit MaxPoolWithArgmaxBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~MaxPoolWithArgmaxBaseTiling() override
    {
    }

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
}  // namespace optiling

#endif