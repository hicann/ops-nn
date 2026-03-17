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
 * \file embedding_tiling_simt.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_TILING_SIMT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_TILING_SIMT_H
#include <cmath>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "error_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_api/runtime2_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

BEGIN_TILING_DATA_DEF(EmbeddingTilingDataSimtTwoDim)
TILING_DATA_FIELD_DEF(int16_t, needCoreNum);
TILING_DATA_FIELD_DEF(int16_t, threadNum);
TILING_DATA_FIELD_DEF(int32_t, gatherDimSize);
TILING_DATA_FIELD_DEF(int32_t, innerSize);
TILING_DATA_FIELD_DEF(int32_t, perCoreElements);
TILING_DATA_FIELD_DEF(int32_t, lastCoreElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Embedding, EmbeddingTilingDataSimtTwoDim)
struct EmbeddingCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class EmbeddingTilingBase : public TilingBaseClass {
public:
    explicit EmbeddingTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~EmbeddingTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
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
    void Reset();

private:
    inline ge::graphStatus GetXInfoAndCheck();
    inline ge::graphStatus GetIndicesInfoAndCheck();
    void ShowBaseTilingData();
    ge::graphStatus MargeAxis();
    ge::graphStatus SimtTwoDimTiling();
    int64_t XDtypeImprove();
    bool IsSimtTwoDim();
    int64_t aivNum_;
    const char *opName_ = "";
    EmbeddingTilingDataSimtTwoDim simtTwoDimTilingData_;
#ifdef DAVID_FPGA
    int64_t threadNum_ = 128;
#else
    int64_t threadNum_ = 2048;
#endif
    gert::Shape xShape_;
    gert::Shape indicesShape_;
    ge::DataType xDtype_;
    ge::DataType indicesDtype_;
    int64_t axis_ = 0;
    int64_t batchDims_ = 0;
    int64_t ySize_ = 1;
    int32_t improveDtypeSize_ = 0;
    int32_t indicesDtypeSize_ = 0;
    int64_t gatherDimSize_ = 0;
    int64_t batchSize_ = 1;
    int64_t outerSize_ = 1;
    int64_t gatherSize_ = 0;
    int64_t innerSize_ = 1;
    int64_t ubSize_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t inputBatchDims_ = 0;
    bool negativeIndexSupport_ = false;
    bool supportOutOfBoundIndex_ = false;
    int32_t tilingMode_ = 0;
    int32_t ubBlockSize_ = 32;
    int32_t vRegSize_ = 256;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_H