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
 * \file adaptive_max_pool2d_tiling_base.h
 * \brief tiling base imply for adaptive_max_pool2d
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_MAX_POOL2D_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_MAX_POOL2D_TILING_BASE_H_

#include <array>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

using namespace std;

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;    
const int HW_DIMS = 2;
const uint32_t H_DIM = 0;
const uint32_t W_DIM = 1;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;
const uint32_t BLOCK_LEN_FP32 = 8;
const uint32_t BLOCK_LEN_FP16 = 16;

struct AdaptiveMaxPool2dCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    size_t sysWorkspaceSize = 0;
};

struct InputInfo {
    uint64_t coreNum{0};
    uint64_t ubSizePlatForm{0};
    ge::DataType xDtype{ge::DT_FLOAT};
    ge::DataType indicesDtype{ge::DT_INT32};
    uint64_t N{0};
    uint64_t C{0};
    uint64_t Hi{0};
    uint64_t Wi{0};
    uint64_t Ho{0};
    uint64_t Wo{0};
};

struct CalculateInfo {
    uint64_t useCoreNum{0};
    uint64_t totalIdx{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t ncFactor{0};
    uint64_t hoFactor{0};
    uint64_t woFactor{0};
    uint64_t ncOuter{0};
    uint64_t hoOuter{0};
    uint64_t woOuter{0};
    uint64_t ncTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t kernelHMax{0};
    uint64_t kernelWMax{0};
};

BEGIN_TILING_DATA_DEF(AdaptiveMaxPool2dTilingData)
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, C);
TILING_DATA_FIELD_DEF(int64_t, Hi);
TILING_DATA_FIELD_DEF(int64_t, Wi);
TILING_DATA_FIELD_DEF(int64_t, Ho);
TILING_DATA_FIELD_DEF(int64_t, Wo);
TILING_DATA_FIELD_DEF(int64_t, coreNums);
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, totalIdx);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ncFactor);
TILING_DATA_FIELD_DEF(int64_t, hoFactor);
TILING_DATA_FIELD_DEF(int64_t, woFactor);
TILING_DATA_FIELD_DEF(int64_t, ncOuter);
TILING_DATA_FIELD_DEF(int64_t, hoOuter);
TILING_DATA_FIELD_DEF(int64_t, woOuter);
TILING_DATA_FIELD_DEF(int64_t, ncTail);
TILING_DATA_FIELD_DEF(int64_t, hoTail);
TILING_DATA_FIELD_DEF(int64_t, woTail);

TILING_DATA_FIELD_DEF(int64_t, threadNums);
TILING_DATA_FIELD_DEF(int64_t, blockNums);
TILING_DATA_FIELD_DEF(int64_t, kMaxSizeH);
TILING_DATA_FIELD_DEF(int64_t, kMaxSizeW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool2d, AdaptiveMaxPool2dTilingData);


class AdaMaxPool2dBaseTiling : public TilingBaseClass {
public:
    explicit AdaMaxPool2dBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    ~AdaMaxPool2dBaseTiling() override
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
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    size_t sysWorkspaceSize_ = 0;
    InputInfo input_;
    CalculateInfo calInfo_;
};
} // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_MAX_POOL2D_TILING_BASE_H_
