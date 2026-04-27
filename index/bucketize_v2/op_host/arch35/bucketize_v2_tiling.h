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
 * \file bucketize_v2_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_BUCKETIZE_V2_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_BUCKETIZE_V2_TILING_BASE_H_

#include <array>

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_api/op_util.h"

namespace optiling {
struct BucketizeV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class BucketizeV2BaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BucketizeV2BaseTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

    ~BucketizeV2BaseTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetShapeAndDtype();
    ge::graphStatus GetAttrsInfo();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    bool IsInvalidType(const ge::DataType dataType);
    int64_t GetUpLog2(int64_t n);

public:
    ge::DataType xDtype_ = ge::DataType::DT_FLOAT;
    ge::DataType yDtype_ = ge::DataType::DT_INT32;
    ge::DataType boundDtype_ = ge::DataType::DT_FLOAT;
    int64_t xDtypeSize_ = 0;
    int64_t boundDtypeSize_ = 0;
    int64_t yDtypeSize_ = 0;
    int64_t xSize_ = 0;
    int64_t boundSize_ = 0;
    int64_t ySize_ = 0;
    int64_t coreNum_ = 1;
    uint32_t ubSize_ = 0;
    bool right_ = false;
    bool outInt32_ = false;
};

} // namespace optiling

#endif