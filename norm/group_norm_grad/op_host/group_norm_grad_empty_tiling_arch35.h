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
 * \file group_norm_grad_empty_tiling_arch35.h
 * \brief
 */

#pragma once

#include "group_norm_grad_tiling.h"

namespace optiling {
class GroupNormGradEmptyTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GroupNormGradEmptyTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    // 1、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 2、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 3、二分折叠reduce计算
    bool IsCapable() override;
    // 4、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 6、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 8、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void CalcRowsAndCols(gert::Shape& gammaShape);
    void CalcUsedCoreNumGamma();
    ge::graphStatus CheckInputAndOutput();
    ge::graphStatus CheckShapeAndType();
    ge::graphStatus ParamsCheck();
    ge::graphStatus InputCheck(gert::Shape& dyShape);
    void SetTilingData();
    void PrintTilingData() const;
    ge::graphStatus CalcuTilingData();
    uint64_t NearestLowerPowerOfTwo(int32_t tmp);
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);

private:
    const char* opName = "GroupNormGrad";
    GroupNormGradEmptyTilingData tilingData;
    uint32_t aivCoreNum_;
    uint64_t rows_; 
    uint64_t cols_; 
    uint64_t usedCoreNumDG_;
    uint64_t colsPerCoreDG_;
    uint64_t colsPerUBDG_;
    uint64_t tailUbCols_;
    uint64_t lastCoreBlockCount_;
    uint64_t lastCoreTailUbCols_;
    uint64_t coreUbBlockCount_;
    uint64_t colsLastCoreDG_;
    uint64_t ubSize_;
    
    uint32_t isMultiColset_;
    uint32_t sysWorkspaceSize_ = 0;
    int64_t workSpaceSize_ = 0;
    ge::DataType tTypeStr_ = ge::DT_UNDEFINED; // T type: x, dy, dx
    ge::DataType uTypeStr_ = ge::DT_UNDEFINED; // U type: mean, rstd, gamma, dgamma, dbeta
    uint32_t modeKey_ = -1;
    int64_t N_ = 0;
    int64_t C_ = 0;
    int64_t G_ = 0;

    int64_t CPerG_ = 0;
    uint32_t vectorLen_ = 0;
};
} // namespace optiling
