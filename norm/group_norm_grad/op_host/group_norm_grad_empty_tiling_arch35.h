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
// empty 模板 tiling：空tensor场景（x总元素数为0且gamma非空），dgamma/dbeta输出全0
// 继承 GroupNormGradTilingBase，注册优先级高于 mix，由 IsCapable 判断是否接管
class GroupNormGradEmptyTiling : public GroupNormGradTilingBase {
public:
    explicit GroupNormGradEmptyTiling(gert::TilingContext* context) : GroupNormGradTilingBase(context) {}

protected:
    // 1、获取INPUT/OUTPUT/ATTR信息（空tensor宽松版：允许dim为0）
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、空tensor场景自判（regbase 且 x总元素==0 且 gamma非空）
    bool IsCapable() override;
    // 4、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 6、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、计算Workspace大小
    ge::graphStatus GetWorkspaceSize() override;
    // 8、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    // 空tensor宽松版输入校验（无逐dim大于等于1检查，允许空维度）
    ge::graphStatus InputCheck(gert::Shape& dyShape) override;
    ge::graphStatus ParamsCheck() override;
    void CalcRowsAndCols(gert::Shape& gammaShape);
    void CalcUsedCoreNumGamma();
    ge::graphStatus CheckInputAndOutput();
    ge::graphStatus CheckShapeAndType();
    void SetTilingData();
    void PrintTilingData() const;
    ge::graphStatus CalcuTilingData();
    uint64_t NearestLowerPowerOfTwo(int32_t tmp);
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);

private:
    const char* opName = "GroupNormGrad";
    GroupNormGradEmptyTilingData tilingData;
    uint32_t modeKey_ = -1;
    int64_t workSpaceSize_ = 0;
    int64_t xShapeDimValue_ = 1;     // x 总元素数（GetShapeAttrsInfo 阶段计算）
    int64_t gammaShapeDimValue_ = 1; // gamma 总元素数（GetShapeAttrsInfo 阶段计算）
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

    uint32_t isMultiColset_;
};
} // namespace optiling
