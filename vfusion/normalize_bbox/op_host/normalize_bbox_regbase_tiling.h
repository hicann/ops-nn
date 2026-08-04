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
 * \file normalize_bbox_regbase_tiling.h
 * \brief NormalizeBBox arch35 tiling (7-step TilingBaseClass)
 */
#pragma once

#include <string>
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_key.h"
#include "../op_kernel/arch35/normalize_bbox_tiling_data.h"
#include "../op_kernel/arch35/normalize_bbox_tiling_key.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

class NormalizeBBoxTilingForRegbase : public TilingBaseClass {
public:
    explicit NormalizeBBoxTilingForRegbase(gert::TilingContext* context)
        : TilingBaseClass(context), opName_(context->GetNodeName())
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

    void PrintTilingData();

private:
    ge::graphStatus GetDtypeAndAttr();
    ge::graphStatus ValidateShapes(const gert::Shape& boxesGeShape, const gert::Shape& shapeHwGeShape,
                                   uint64_t& boxesRank);
    static uint64_t ComputeNum(const gert::Shape& boxesGeShape, uint64_t boxesRank, bool reversedBox,
                               const std::string& opName, ge::graphStatus& status);
    void ComputeTileLen();
    void SplitByBatch(uint64_t batch);
    void SplitByNum(uint64_t num);

    const std::string opName_;
    uint64_t ubSize_{0};
    uint64_t totalCoreNum_{0};
    ge::DataType boxesDType_{ge::DT_FLOAT16};
    uint32_t boxesDtypeSize_{0};
    bool reversedBox_{false};
    ::NormalizeBBoxTilingData tilingData_;
};

} // namespace optiling
