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
 * \file rotate_quant_tiling.h
 * \brief
 */
#ifndef __OP_HOST_ROTATE_QUANT_TILING_H__
#define __OP_HOST_ROTATE_QUANT_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../../op_kernel/rotate_quant_tiling_data.h"

namespace Ops {
namespace NN {
namespace RotateQuant {

struct RotateQuantCompileInfo {
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    uint64_t l2Size{0};
    uint64_t l0ASize{0};
    uint64_t l0BSize{0};
    uint64_t l0CSize{0};
    uint64_t aicNum{0};
    uint64_t aivNum{0};
    uint64_t cubeFreq{0};
};

struct RotateQuantInfo {
    int64_t usedCoreNum = 0;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t numBlocks = 0;
    int32_t stepLoop = 0;
    bool initFlag = false;
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType yDtype = ge::DT_INT8;
    const char *opName = "RotateQuant";
};

class RotateQuantTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RotateQuantTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context){
        InitCompileInfo();
    };
    ~RotateQuantTiling() override = default;

protected:
    bool IsCapable() override {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeAttrs();
    ge::graphStatus AnalyzeShapes();
    void PrintTilingData();
    ge::graphStatus CaclVecTilingData();
    bool SetMatmulTiling();
    bool SetRotateQuantTiling();
    uint32_t CalcMaxHandleRowsPerUb(uint32_t ubSize);
    ge::graphStatus CalcStepLoop(uint32_t maxHandleRowsPerUb, uint32_t loopPerHeadCore,
                                  uint32_t loopPerHeadCubeCore, uint32_t& stepLoop);

    RotateQuantOpt::RotateQuantTilingData tilingData_;
    RotateQuantInfo inputParams_;
    RotateQuantCompileInfo compileInfo_;
    size_t tilingDataSize_ = 0;
};
} // namespace RotateQuant
} // namespace NN
} // namespace Ops
#endif  // __OP_HOST_ROTATE_QUANT_TILING_H__
