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
 * \file batch_norm_grad_v3_tiling_infer_base.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_BASE_TILING_H_
#define BATCH_NORM_GRAD_V3_BASE_TILING_H_

#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
using namespace ge;

// 按照 dy、weight、var 数据类型区分模板
constexpr int64_t TILINGKEY_INFER_FULLLOAD_BASE = 10001;
constexpr int64_t TILINGKEY_INFER_SPLITLOAD_BASE = 20001;
constexpr int64_t TILINGKEY_TRAIN_FULLLOAD_BASE = 30001;
constexpr int64_t TILINGKEY_TRAIN_SPLITLOAD_BASE = 40001;
constexpr int64_t DTYPE_CHECK_NUM = 3;
constexpr int64_t DTYPE_DX_OFFSET = 0;
constexpr int64_t DTYPE_WEIGHT_OFFSET = 1;
constexpr int64_t DTYPE_RUNNINGVAR_OFFSET = 2;
static std::vector<std::array<ge::DataType, DTYPE_CHECK_NUM>> validInputDtypes = {
    {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},       // 1
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16}, // 2
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16},          // 3
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},     // 4
    {ge::DataType::DT_BF16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},        // 5
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT},   // 6
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT},         // 7
};

constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_NUM_5 = 5;

constexpr int64_t INPUT_OUTPUT_NUM = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

constexpr int64_t PARAM_INPUT_WEIGHT_INDEX = 2;
constexpr int64_t PARAM_INPUT_RUNNINGVAR_INDEX = 4;
constexpr int64_t PARAM_OUTPUT_DX_INDEX = 0;
constexpr int64_t PARAM_ATTRS_EPSILON_INDEX = 1;

constexpr float DEFAULT_EPSILON = 1e-5;
constexpr int64_t VL_DOUBLE = 2;
constexpr int64_t FLOAT_BLOCK_SIZE = 8;
constexpr int64_t HALF_BLOCK_SIZE = 16;

// 框架侧占位可以只预留32B（ttk正常），debugTool执行时需要预留16M
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;

class BatchNormGradV3Base : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormGradV3Base(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~BatchNormGradV3Base() override = default;

    void Reset(gert::TilingContext* context) override
    {
        Ops::NN::Optiling::TilingBaseClass::Reset(context);
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
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        return 0;
    }
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    void Reset();
    void CalcBasicInfo();
    ge::graphStatus GetDyInfo();
    ge::graphStatus GetWeightRunningVarDxInfo();
    ge::graphStatus CheckInputValid();
    int64_t GetAlignValue(int64_t value);

protected:
    const char* opName_ = "BatchNormGradV3Base";

    ge::Format dyFormat_;

    int64_t needCoreNum;

    int64_t blockSize_;
    int64_t vlFp32_;
    int64_t vlFp16_;

    int64_t bytesPerDy_;
    int64_t bytesPerWeight_;
    int64_t bytesPerRunningVar_;

    int64_t fusedALen_;
    int64_t fusedB0Len_;
    int64_t fusedB1Len_;
    int64_t bAlign;
    int64_t fusedB1Align;

    int64_t dyDimNum_;
    int64_t weightDimNum_;
    int64_t runningVarDimNum_;
    int64_t dxDimNum_;

    int64_t weightDimLen_;
    int64_t runningVarDimLen_;

    int64_t tilingKeyOffset_;

    bool isTraining;
    float epsilon_;

    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;

    ge::DataType dyDtype_{ge::DataType::DT_FLOAT};
    ge::DataType weightDtype_{ge::DataType::DT_FLOAT};
    ge::DataType runningVarDtype_{ge::DataType::DT_FLOAT};
    ge::DataType dxDtype_{ge::DataType::DT_FLOAT};
};

} // namespace optiling

#endif