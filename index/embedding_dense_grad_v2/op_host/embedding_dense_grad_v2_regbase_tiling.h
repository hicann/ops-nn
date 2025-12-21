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
 * \file embedding_dense_grad_v2_regbase_tiling.h
 * \brief
 */
#pragma once

#include <string>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_key.h"
#include "index/embedding_dense_grad_v2/op_kernel/v35/embedding_dense_grad_v2_struct.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

class EmbeddingDenseGradV2ForRegBase : public TilingBaseClass
{
public:
    explicit EmbeddingDenseGradV2ForRegBase(gert::TilingContext* context)
        : TilingBaseClass(context), opName_(context->GetNodeName())
    {}

protected:
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
    int64_t totalCoreNum_{0};
    int64_t scatterAxis_{1};
    int64_t elewiseAxis_{0};
    uint32_t scatterFactor_{0};
    uint32_t elewiseFactor_{0};
    ge::DataType gradDType_{ge::DT_FLOAT};
    ge::DataType indicesDtype_{ge::DT_INT32};
    uint32_t gradDtypeSize_{0};
    uint32_t indicesDtypeSize_{0};
    uint32_t elewiseNormalCount_{0};
    uint32_t elewiseTailCount_{0};
    uint64_t blockLastNum_{0};
    uint64_t normalBlockLastLoop_{0};
    uint64_t tailBlockLastLoop_{0};

    // Op attr
    int64_t numWeights_{0};
    uint64_t embeddingDim_{0};
    int64_t paddingIdx_{0};
    bool scaleGrad_{false};

    ge::graphStatus VerifyIndicesAndPosIdx();
    bool DoUBTilingSingle(int64_t avaliableUbSize, uint32_t elewiseAligned, int64_t resBufSize);
    void FinalizeProcessOpTiling();
    void SetTilingData(
        int64_t scatterDimLoopNum, int64_t elewiseDimLoopNum, int64_t elewiseDimOuterLoopNum,
        int64_t normalBlockLoopNum, int64_t tailBlockLoopNum);
};

} // namespace optiling