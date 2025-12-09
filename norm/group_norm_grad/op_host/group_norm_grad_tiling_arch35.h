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
 * \file group_norm_grad_regbase_tiling.h
 * \brief
 */

#pragma once

#include "group_norm_grad_tiling.h"
using namespace Ops::NN::Optiling;

namespace optiling {
class GroupNormGradRegBaseTiling : public TilingBaseClass {
public:
    explicit GroupNormGradRegBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
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
    static constexpr uint32_t CACHE_BUFF_SIZE = 256;
    static constexpr uint32_t UPPER_CARRYING_LIMIT = 8000;
    static constexpr uint32_t MAX_C_SIZE = 240000;
    static constexpr uint32_t WORKSPACE_COPIES = 2;
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr uint32_t BINARY_ADD_COEF = 2;
    static constexpr uint32_t UB_COPIES_4 = 4; // dbeta, dgamma, ds, gamma
    static constexpr uint32_t UB_COPIES_3 = 3; // x, dy, dx
    static constexpr uint32_t UB_COPIES_2 = 2; // rstd, mean or (x, dy)
    static constexpr uint32_t SPLIT_COUNT = 2;
    static constexpr int64_t MODE_0 = 100; // g_full_load
    static constexpr int64_t MODE_1 = 200; // c_full_load
    static constexpr int64_t MODE_2 = 300; // recompute
    static constexpr int64_t MODE_3 = 400; // small NxG
    static constexpr int64_t STAGE2_MODE_1 = 1;
    static constexpr int64_t STAGE2_MODE_2 = 2;

    uint32_t GetTypeSize(ge::DataType dtypeStr) const;
    ge::graphStatus BlockTiling();
    ge::graphStatus UbTiling();
    ge::graphStatus Mode2UbTiling();
    void CalBinaryParams(int64_t reduceNum, int64_t& binaryQuotient, uint32_t& binaryK, uint32_t& binaryLastNum);
    ge::graphStatus ParamsCheck();
    ge::graphStatus InputCheck(gert::Shape& dyShape);
    void SetTilingData();
    void SetBaseTilingData();
    void SetReduceTilingData();
    void SetKernel2TilingData();
    void PrintTilingData() const;
    ge::graphStatus Stage2Mode2Tiling();
    ge::graphStatus Stage2Tiling();

private:
    const char* opName = "GroupNormGrad";
    GroupNormGradRegBaseTilingData tilingData;
    uint64_t ubSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    uint32_t blockSize_ = 0;
    int64_t workSpaceSize_ = 0;
    uint32_t reserveSpace_ = 0;
    ge::DataType tTypeStr_ = ge::DT_UNDEFINED; // T type: x, dy, dx
    ge::DataType uTypeStr_ = ge::DT_UNDEFINED; // U type: mean, rstd, gamma, dgamma, dbeta
    uint32_t tTypeBytes_ = 0;
    uint32_t modeKey_ = -1;
    uint32_t stage0CoreUsed_ = 0;
    uint32_t stage0TailCore_ = 0;
    uint32_t stage1CoreUsed_ = 0;
    int64_t N_ = 0;
    int64_t C_ = 0;
    int64_t HxW_ = 1;
    int64_t G_ = 0;
    int64_t NxC_ = 0;
    int64_t NxG_ = 0;
    int64_t CPerG_ = 0;
    int64_t clrBlockSize_ = 0;
    uint32_t clrBlockNum_ = 0;
    int64_t clrTailBlockSize_ = 0;
    uint32_t stage0TaskNumPerCore_ = 0;
    uint32_t stage0TaskNumPerTailCore_ = 0;
    uint32_t taskNumPerCore_ = 0;
    uint32_t taskNumPerTailCore_ = 0;
    uint32_t tailCore_ = 0;
    uint32_t mode0UbCapGNum_ = 0;
    uint32_t mode1UbCapCNum_ = 0;
    uint32_t mode2MainLoopCnt_ = 0;
    uint32_t mode2FoldLoopCnt_ = 0;
    uint32_t mode2OneLoopSize_ = 0;
    uint32_t mode2MainTail_ = 0;
    uint32_t mode2FoldTail_ = 0;
    uint32_t mode2UbCapEle_ = 0;
    uint32_t mode2UbIterNum_ = 0;
    uint32_t mode2UbTailNum_ = 0;
    int64_t binaryAddQuotient_ = 0;
    uint32_t binaryAddK_ = 0;
    uint32_t binaryAddLastNum_ = 0;
    int64_t binaryCGQuotient_ = 0;
    uint32_t binaryCGK_ = 0;
    uint32_t binaryCGLastNum_ = 0;
    uint32_t cFactor_ = 0;
    uint32_t cFactorAlign_ = 0;
    uint32_t cBlockFactor_ = 0;
    uint32_t cTailBlockFactor_ = 0;
    uint32_t stage2CoreUsed_ = 0;
    uint32_t stage2BinaryAddQuotient_ = 0;
    uint32_t stage2BinaryAddK_ = 0;
    int64_t reduceNCnt_ = 0;
    uint32_t stage2Mode_ = 1;
    uint32_t cNumMode2PerCore_ = 0;
    uint32_t tailcNumMode2PerCore_ = 0;
    int64_t stage2LoopCnt_ = 0;
    int64_t stage2FactorN_ = 0;
    uint32_t vectorLen_ = 0;
    bool dxIsRequire_ = false;
    bool dgammaIsRequire_ = false;
    bool dbetaIsRequire_ = false;
};
} // namespace optiling
