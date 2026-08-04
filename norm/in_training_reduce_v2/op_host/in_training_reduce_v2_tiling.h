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
 * \file in_training_reduce_v2_tiling.h
 * \brief
 */

#ifndef IN_TRAINING_REDUCE_V2_TILING_H
#define IN_TRAINING_REDUCE_V2_TILING_H

#include <cmath>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../op_kernel/arch35/in_training_reduce_v2_tiling_data.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {
static constexpr uint64_t IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_PRIORITY = 9000;

constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_SUM_INDEX = 0;
constexpr uint32_t OUTPUT_SQUARE_SUM_INDEX = 1;

struct INTrainingReduceV2CompileInfo {
    uint64_t coreNum;      // 系统核数
    uint64_t ubSize;       // UB 空间
    uint32_t vectorLength; // 256
    uint64_t ubBlockSize;  // 32B，UB 的字节对齐单位
};

class INTrainingReduceV2RegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit INTrainingReduceV2RegbaseTilingBase(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        a1 = 0;
        a0 = 0;
        r = 0;
        vlfp32 = 0;
        vectorLength = 0;
        ubBlockSize = 0;
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckShapeAllNotNegative(gert::Shape& shape);
    ge::graphStatus ParseShapeByFormat();

protected:
    int64_t a1{0};
    int64_t a0{0};
    int64_t r{0};
    int64_t vlfp32{0};
    int64_t vectorLength{0};
    int64_t ubBlockSize{0}; // 用于在 UB 上进行 32B 的字节对齐

    int64_t blockNum_{1};

    ge::DataType dataType{ge::DT_UNDEFINED};
    ge::Format format{ge::FORMAT_ND};
    gert::Shape xStorageShape;
};

class INTrainingReduceV2ARFullReduceTiling : public INTrainingReduceV2RegbaseTilingBase {
public:
    explicit INTrainingReduceV2ARFullReduceTiling(gert::TilingContext* context)
        : INTrainingReduceV2RegbaseTilingBase(context)
    {}
    ~INTrainingReduceV2ARFullReduceTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        INTrainingReduceV2RegbaseTilingBase::Reset(context);
        blockNum_ = 0;
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    // sub-R 分块 tiling（R 超单次 UB 容量，DESIGN §6.3 路 A）
    bool DoSubRTiling(uint64_t rAlign, uint64_t binAddQuotient, int64_t elemSize);

    int64_t binaryAddQuotient;
    INTrainingReduceV2ARFullReduceTilingData td_;
};

} // namespace optiling
#endif // IN_TRAINING_REDUCE_V2_TILING_H
