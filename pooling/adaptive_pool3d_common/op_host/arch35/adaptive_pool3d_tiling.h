/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_pool3d_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (51) USING BLANK LINES.
 * 
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_POOL3D_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_POOL3D_TILING_H_

#include <array>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"

using namespace std;

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

struct BaseInput {
    uint64_t coreNum{0};
    uint64_t ubSize{0};
    ge::DataType xDtype{ge::DT_FLOAT};
    ge::DataType indicesDtype{ge::DT_INT32};
    uint64_t nIn{0};
    uint64_t cIn{0};
    uint64_t dIn{0};
    uint64_t hIn{0};
    uint64_t wIn{0};
    uint64_t dOut{0};
    uint64_t hOut{0};
    uint64_t wOut{0};
};

struct AdaptivePool3dCompileInfo {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
};

class AdaptivePool3dBaseTiling : public TilingBaseClass {
public:
    explicit AdaptivePool3dBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~AdaptivePool3dBaseTiling() override
    {}

    BaseInput input_;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
};

}// namespace optiling

#endif