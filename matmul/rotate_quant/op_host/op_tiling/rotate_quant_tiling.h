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

namespace optiling {
using namespace RotateQuantOpt;
using Ops::NN::Optiling::TilingBaseClass;

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
public:
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

class RotateQuantTiling : public TilingBaseClass {
public:
    explicit RotateQuantTiling(gert::TilingContext* context) : TilingBaseClass(context){
        InitCompileInfo();
    };
    ~RotateQuantTiling() override = default;

protected:
    bool IsCapable() override {
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

    RotateQuantTilingData tilingData_;
    RotateQuantInfo inputParams_;
    RotateQuantCompileInfo compileInfo_;
    size_t tilingDataSize_ = 0;
};
}  // namespace optiling
#endif  // __OP_HOST_ROTATE_QUANT_TILING_H__
