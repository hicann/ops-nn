/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_to_dense.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSE_TO_DENSE_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSE_TO_DENSE_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "../op_kernel/arch35/sparse_to_dense_struct.h"

namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;


struct SparseToDenseCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

class SparseToDenseTiling : public TilingBaseClass
{
public:
    explicit SparseToDenseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckInputDtype();
    void SetTilingData();
    
    template <typename T>
    ge::graphStatus GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape);

private:
    int64_t numValues_ = 0;
    int64_t outSize_ = 1;
    int64_t numDims_ = 0;
    int64_t normCoreHandleDefaultValues_ = 0;
    int64_t defaultUbFactor_ = 0;
    int64_t normBlockTailLoopSize_ = 0;
    int64_t tailBlockTailLoopSize_ = 0;
    int64_t normBlockLoop_ = 0;
    int64_t tailBlockLoop_ = 0;
    int64_t normCoreHandleSparses_ = 0;
    int64_t tailCoreHandleSparses_ = 0;
    int64_t defaultValueUsedCoreNum_ = 0;
    int64_t sparseUsedCoreNum_ = 0;
    bool validateIndices_ = true;
    bool isNeedSyncall_ = true;

    int64_t isValuesScalar_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t useCoreNum_ = 0;
    int64_t tilingKey_ = 0;
    int64_t ubSize_ = 0;
    int64_t indicesDimNum_ = 0;
    int64_t indicesFirstDim_ = 0;
    int64_t indicesLastDim_ = 0;
    int64_t outputShapeDimNum_ = 0;
    int64_t valueDimNum_ = 0;
    int64_t valueSize_ = 0;
    int64_t defaultValueDimNum_ = 0;
    int64_t defaultValueSize_ = 0;

    gert::Shape outputShape_;
    ge::DataType valuesDtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    SparseToDenseTilingData tilingData_;
    const char* opName_ = "SparseToDense";
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSE_TO_DENSE_TILING_H_