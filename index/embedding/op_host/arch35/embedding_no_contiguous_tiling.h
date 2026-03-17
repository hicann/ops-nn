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
 * \file embedding_no_contiguous_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_NO_CONTIGUOUS_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_NO_CONTIGUOUS_TILING_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "error_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

BEGIN_TILING_DATA_DEF(EmbeddingNoContiguousTilingData)
TILING_DATA_FIELD_DEF(int64_t, threadNum);         // 使用线程数
TILING_DATA_FIELD_DEF(int64_t, ySize);             // 输出y的数据量大小
TILING_DATA_FIELD_DEF(int64_t, gatherSize);        // 输入x的0轴大小
TILING_DATA_FIELD_DEF(int64_t, innerSize);         // 输入x的尾轴大小
TILING_DATA_FIELD_DEF(int64_t, indicesDim1Size);   // 输入indices的尾轴大小
TILING_DATA_FIELD_DEF(int64_t, xDim0Stride);       // 输入x第0维stride
TILING_DATA_FIELD_DEF(int64_t, xDim1Stride);       // 输入x第1维stride
TILING_DATA_FIELD_DEF(int64_t, indicesDim0Stride); // 输入indices第0维stride
TILING_DATA_FIELD_DEF(int64_t, indicesDim1Stride); // 输入indices第1维stride
TILING_DATA_FIELD_DEF(int64_t, needCoreNum); 
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Embedding_101, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_102, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_104, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_108, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_111, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_112, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_114, EmbeddingNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(Embedding_118, EmbeddingNoContiguousTilingData)

class EmbeddingNoContiguousTiling : public TilingBaseClass
{
public:
    explicit EmbeddingNoContiguousTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    // 执行顺序 2 -> 1 -> 0 -> 3 -> 4 -> 6 -> 7 -> 5 -> 8
    // 0、模板条件
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
    // 8、Dump Tiling数据
    void DumpTilingInfo() override;

private:
    static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
    static constexpr int64_t SMALL_CASE_THREAD_NUM = 128;
    static constexpr int64_t NUM_TWO = 2;

    inline bool ParamTypeIsInvalid(ge::DataType& x);
    ge::graphStatus CheckInAndOutDtype();
    ge::graphStatus CheckOutShape();
    void CalcuCore();
    bool IsContiguous(const gert::Shape& xShape, const gert::Stride& xStride);
    ge::graphStatus GetContiguousTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx, bool isOut);
    ge::graphStatus GetTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx, bool isOut = false);
    void SetTilingData();
    bool IsEnableInt64(gert::Shape shape, gert::Stride stride);

private:
    int64_t ySize_ = 1;
    int64_t indicesDim1Size_ = 0;
    int64_t gatherSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t xDim0Stride_ = 1;
    int64_t xDim1Stride_ = 1;
    int64_t indicesDim0Stride_ = 1;
    int64_t indicesDim1Stride_ = 1;

    int64_t threadNum_ = 2048;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t enableInt64_ = 0;
    int64_t indicesDtypeSize_ = 0;
    int64_t xDtypeSize_ = 0;
    int64_t needCoreNum_ = 1;
    int64_t perCoreElements_ = 1;
    int64_t lastCoreElements_ = 1;

    gert::Shape xShape_;
    gert::Shape indicesShape_;
    gert::Shape yShape_;

    gert::Stride xStride_;
    gert::Stride indicesStride_;
    gert::Stride yStride_;

    ge::DataType xDtype_;
    ge::DataType indicesDtype_;

    const char* opName_ = "Embedding";
    EmbeddingNoContiguousTilingData m_tilingData_;
};
} // namespace optiling
#endif // EMBEDDING_NO_CONTIGUOUS_TILING_H
