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
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"

using namespace std;

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
constexpr int64_t MAX_INT32 = 2147483647;
constexpr uint64_t MAX_UINT32 = 4294967295;
constexpr int64_t MAX_THREAD_NUM = 1024;
constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
constexpr uint64_t DIM_N = 0;
constexpr uint64_t DIM_C = 1;
constexpr uint64_t DIM_D = 2;
constexpr uint64_t DIM_H = 3;
constexpr uint64_t DIM_W = 4;
constexpr uint64_t OUTPUTSIZE_DIMW = 2;
constexpr uint64_t OUTPUT_DIM_MAX = 3;
constexpr uint64_t DIM_NUM_FIVE = 5;
constexpr uint64_t DIM_NUM_FOUR = 4;
constexpr int64_t DTYPE_INT32 = 3;
constexpr int64_t DTYPE_INT64 = 9;
constexpr int64_t ONE_DIM = 1;
constexpr int64_t NONE_DIM = 0;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t KERNEL_CALC_COUNT_THERSHOLD = 10000;

struct BaseInput {
    uint64_t coreNum{0};
    uint64_t ubSize{0};
    ge::DataType xDtype{ge::DT_FLOAT};
    ge::DataType indicesDtype{ge::DT_INT32};
    ge::Format dataFormat{ge::Format::FORMAT_NDHWC};
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
    ge::graphStatus GetAndCheckIndicesDtype();
    ge::graphStatus GetAndCheckDataFormat();
    uint64_t CalKernelSizeOneDimMax(uint64_t inSize, uint64_t outSize);
};

}// namespace optiling

#endif