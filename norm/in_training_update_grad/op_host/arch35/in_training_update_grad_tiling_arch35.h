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
 * \file in_training_update_grad_tiling_arch35.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_IN_TRAINING_UPDATE_GRAD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_IN_TRAINING_UPDATE_GRAD_TILING_H

#include <cmath>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "norm/in_training_update_grad/op_kernel/arch35/in_training_update_grad_tiling_data.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {
static constexpr uint64_t IN_UG_REDUCE_EMPTY_PRIORITY = 5000;
static constexpr uint64_t IN_UG_FULL_LOAD_PRIORITY = 9000;
static constexpr uint64_t IN_UG_STREAM_PRIORITY = 15000;

struct InTrainingUpdateGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    uint32_t vectorLength = 0; // vector register width in bytes (256 on arch35)
    uint64_t ubBlockSize = 0;  // 32B
};

class InTrainingUpdateGradTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit InTrainingUpdateGradTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus CheckDtype();

protected:
    int64_t numN_{0};
    int64_t numD_{0};
    int64_t numC1_{0};
    int64_t numH_{0};
    int64_t numW_{0};
    int64_t numC0_{0};
    int64_t reduceR_{0};      // D * H * W
    int64_t numHW_{0};        // H * W
    int64_t groupNum_{0};     // N * C1
    int64_t blockLenElem_{0}; // H * W * C0
    float epsilon_{1e-6f};

    int64_t vlfp32_{0};
    int64_t vectorLength_{0};
    int64_t ubBlockSize_{0};
    ge::DataType dyDataType_{ge::DT_FLOAT};
    ge::Format format_{ge::FORMAT_ND};
};

class InTrainingUpdateGradReduceEmptyTiling : public InTrainingUpdateGradTilingBase {
public:
    explicit InTrainingUpdateGradReduceEmptyTiling(gert::TilingContext* context)
        : InTrainingUpdateGradTilingBase(context)
    {}
    ~InTrainingUpdateGradReduceEmptyTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    int64_t blockNum_{1};
    InTrainingUpdateGradReduceEmptyTilingData td_;
};

class InTrainingUpdateGradFullLoadTiling : public InTrainingUpdateGradTilingBase {
public:
    explicit InTrainingUpdateGradFullLoadTiling(gert::TilingContext* context) : InTrainingUpdateGradTilingBase(context)
    {}
    ~InTrainingUpdateGradFullLoadTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    int64_t blockNum_{1};
    InTrainingUpdateGradFullLoadTilingData td_;
};

class InTrainingUpdateGradStreamTiling : public InTrainingUpdateGradTilingBase {
public:
    explicit InTrainingUpdateGradStreamTiling(gert::TilingContext* context) : InTrainingUpdateGradTilingBase(context) {}
    ~InTrainingUpdateGradStreamTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    int64_t blockNum_{1};
    InTrainingUpdateGradStreamTilingData td_;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_IN_TRAINING_UPDATE_GRAD_TILING_H
