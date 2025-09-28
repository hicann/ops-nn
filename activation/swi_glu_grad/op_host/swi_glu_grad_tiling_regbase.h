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
 * \file swi_glu_grad_tiling_regbase.h
 * \brief
 */
#pragma once

#include <string>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/arch35/swi_glu_grad_regbase_tiling.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

class GluBaseTiling4RegBase : public TilingBaseClass {
public:
    explicit GluBaseTiling4RegBase(gert::TilingContext* context)
        : TilingBaseClass(context), opName_(context->GetNodeName())
    {}

protected:
    constexpr static int64_t UB_RESERVED_BUFF{0};
    constexpr static int64_t BASE_BLOCK_SIZE{8192};
    constexpr static int64_t MOVE_ALIGN_LIMIT_BYTE{1024};
    constexpr static int64_t BASE_BLOCK_COPY_ALIGN{512};

    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void DumpTilingInfo() override;

private:
    const std::string opName_;
    uint64_t ubSize_{0};
    GluBaseTilingData tilingData_;
    int64_t rowTotalNum_{0};
    int64_t colTotalNum_{0};
    int64_t rowNormalNum_{0};
    int64_t colNormalNum_{0};
    int64_t rowTailNum_{0};
    int64_t colTailNum_{0};
    uint64_t usedCoreNum_{0};
    uint32_t rowTileNum_{0};
    uint32_t colTileNum_{0};
    ge::DataType dataType_{ge::DT_FLOAT};
    uint64_t dataSize_{0};

    bool CalcShapeTo2D(const gert::Shape& inShape, const int64_t dim);
    bool CheckShapeValid(const gert::Shape& gradYShape, const gert::Shape& xShape, const int64_t dim);
    void AutoTiling();
    std::set<int64_t> FindUniqueCut() const;
    uint64_t ComputeTiling(const std::vector<uint32_t>& args) const;
    void SetTilingData();
};
} // namespace optiling