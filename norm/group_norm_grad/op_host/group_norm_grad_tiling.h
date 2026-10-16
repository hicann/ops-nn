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
 * \file group_norm_grad.h
 * \brief
 */
/*!
 * \file group_norm_grad.h
 * \brief tiling of GroupNormGrad op
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_GROUPNORMGRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_GROUPNORMGRAD_H
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormGradTilingData)
TILING_DATA_FIELD_DEF(uint64_t, Tiling_key);               // 0
TILING_DATA_FIELD_DEF(uint64_t, N);                        // 1
TILING_DATA_FIELD_DEF(uint64_t, C);                        // 2
TILING_DATA_FIELD_DEF(uint64_t, HXW);                      // 3
TILING_DATA_FIELD_DEF(uint64_t, G);                        // 4
TILING_DATA_FIELD_DEF(uint64_t, NXG);                      // 5
TILING_DATA_FIELD_DEF(uint64_t, C_G);                      // 6
TILING_DATA_FIELD_DEF(uint64_t, task_num_per_core);        // 7
TILING_DATA_FIELD_DEF(uint64_t, task_num_per_tail_core);   // 8
TILING_DATA_FIELD_DEF(uint64_t, tail_core);                // 9
TILING_DATA_FIELD_DEF(uint64_t, mode1_ub_cap_C_num);       // 10
TILING_DATA_FIELD_DEF(uint64_t, mode1_ub_iter_C_num);      // 11
TILING_DATA_FIELD_DEF(uint64_t, mode1_ub_tail_C_num);      // 12
TILING_DATA_FIELD_DEF(uint64_t, mode2_ub_capacity_ele);    // 13
TILING_DATA_FIELD_DEF(uint64_t, mode2_ub_iteration_num);   // 14
TILING_DATA_FIELD_DEF(uint64_t, mode2_ub_tail_num);        // 15
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);            // 16
TILING_DATA_FIELD_DEF(uint64_t, stage2CoreUsed);           // 17
TILING_DATA_FIELD_DEF(uint64_t, castEleNum);               // 18
TILING_DATA_FIELD_DEF(uint64_t, tailCastNum);              // 19
TILING_DATA_FIELD_DEF(uint64_t, coreBatchParts);           // 20
TILING_DATA_FIELD_DEF(uint64_t, coreBatchPartsTailRepeat); // 21
TILING_DATA_FIELD_DEF(uint64_t, repeatTime4Stage2);        // 22
TILING_DATA_FIELD_DEF(uint64_t, dx_is_require);            // 23
TILING_DATA_FIELD_DEF(uint64_t, dgamma_is_require);        // 24
TILING_DATA_FIELD_DEF(uint64_t, dbeta_is_require);         // 25
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormGrad, GroupNormGradTilingData)

BEGIN_TILING_DATA_DEF(GroupNormGradBaseParams)
TILING_DATA_FIELD_DEF(int64_t, N);                         // 1
TILING_DATA_FIELD_DEF(int64_t, C);                         // 2
TILING_DATA_FIELD_DEF(int64_t, HXW);                       // 3
TILING_DATA_FIELD_DEF(int64_t, G);                         // 4
TILING_DATA_FIELD_DEF(int64_t, clrBlockSize);              // 5
TILING_DATA_FIELD_DEF(int64_t, clrTailBlockSize);          // 6
TILING_DATA_FIELD_DEF(uint32_t, clrBlockNum);              // 7
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerCore);           // 8
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerTailCore);       // 9
TILING_DATA_FIELD_DEF(uint32_t, tailCore);                 // 10
TILING_DATA_FIELD_DEF(uint32_t, stage1CoreUsed);           // 11
TILING_DATA_FIELD_DEF(uint32_t, mode0UbCapGNum);           // 12
TILING_DATA_FIELD_DEF(uint32_t, mode1UbCapCNum);           // 13
TILING_DATA_FIELD_DEF(uint32_t, mode2UbCapEle);            // 14
TILING_DATA_FIELD_DEF(uint32_t, mode2UbIterNum);           // 15
TILING_DATA_FIELD_DEF(uint32_t, mode2UbTailNum);           // 16
TILING_DATA_FIELD_DEF(uint32_t, mode2MainLoopCnt);         // 17
TILING_DATA_FIELD_DEF(uint32_t, mode2FoldLoopCnt);         // 18
TILING_DATA_FIELD_DEF(uint32_t, mode2OneLoopSize);         // 19
TILING_DATA_FIELD_DEF(uint32_t, mode2FoldTail);            // 20
TILING_DATA_FIELD_DEF(uint32_t, workSpaceSize);            // 21
TILING_DATA_FIELD_DEF(int32_t, dxIsRequire);               // 22
TILING_DATA_FIELD_DEF(int32_t, dgammaIsRequire);           // 23
TILING_DATA_FIELD_DEF(int32_t, dbetaIsRequire);            // 24
TILING_DATA_FIELD_DEF(uint32_t, stage0TaskNumPerCore);     // 25
TILING_DATA_FIELD_DEF(uint32_t, stage0TaskNumPerTailCore); // 26
TILING_DATA_FIELD_DEF(uint32_t, stage0TailCore);           // 27
TILING_DATA_FIELD_DEF(uint32_t, stage0CoreUsed);           // 28
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupNormGradBaseParamsOp, GroupNormGradBaseParams)

BEGIN_TILING_DATA_DEF(GroupNormGradReduceParams)
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient); // 1
TILING_DATA_FIELD_DEF(uint32_t, binaryAddK);       // 2
TILING_DATA_FIELD_DEF(uint32_t, binaryAddLastNum); // 3
TILING_DATA_FIELD_DEF(int64_t, binaryCGQuotient);  // 4
TILING_DATA_FIELD_DEF(uint32_t, binaryCGK);        // 5
TILING_DATA_FIELD_DEF(uint32_t, binaryCGLastNum);  // 6
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupNormGradReduceParamsOp, GroupNormGradReduceParams)

BEGIN_TILING_DATA_DEF(GroupNormGradKernel2Params)
TILING_DATA_FIELD_DEF(uint32_t, stage2Mode);
TILING_DATA_FIELD_DEF(uint32_t, cFactor);
TILING_DATA_FIELD_DEF(uint32_t, cBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, cTailBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, stage2CoreUsed);
TILING_DATA_FIELD_DEF(uint32_t, stage2BinaryAddK);
TILING_DATA_FIELD_DEF(uint32_t, cNumMode2PerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailcNumMode2PerCore);
TILING_DATA_FIELD_DEF(int64_t, stage2BinaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, reduceNCnt);    // workSpace中一共有多少个N
TILING_DATA_FIELD_DEF(int64_t, stage2LoopCnt); // 核内需要循环几次
TILING_DATA_FIELD_DEF(int64_t, stage2FactorN); // UB一次处理多少个N
// Stage2Mode2 Use
TILING_DATA_FIELD_DEF(int64_t, cFactorStage2Mode2);
TILING_DATA_FIELD_DEF(int64_t, cLoopMainBlock);
TILING_DATA_FIELD_DEF(int64_t, cTailMainBlock);
TILING_DATA_FIELD_DEF(int64_t, cLoopTailBlock);
TILING_DATA_FIELD_DEF(int64_t, cTailTailBlock);
TILING_DATA_FIELD_DEF(int64_t, nFactorStage2Mode2);
TILING_DATA_FIELD_DEF(int64_t, nLoop);
TILING_DATA_FIELD_DEF(int64_t, nMainFlodCount);
TILING_DATA_FIELD_DEF(int64_t, nTail);
TILING_DATA_FIELD_DEF(int64_t, cacheBufferCount);
TILING_DATA_FIELD_DEF(int32_t, resultCacheId);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupNormGradKernel2ParamsOp, GroupNormGradKernel2Params)

BEGIN_TILING_DATA_DEF(GroupNormGradRegBaseTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GroupNormGradBaseParams, gNGBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(GroupNormGradReduceParams, gNGReduceParams);
TILING_DATA_FIELD_DEF_STRUCT(GroupNormGradKernel2Params, gNGKernel2Params);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormGrad_101, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_102, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_103, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_104, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_105, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_201, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_202, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_203, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_204, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_205, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_301, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_302, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_303, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_304, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_305, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_401, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_402, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_403, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_404, GroupNormGradRegBaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormGrad_405, GroupNormGradRegBaseTilingData)

BEGIN_TILING_DATA_DEF(GroupNormGradEmptyTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDG);
TILING_DATA_FIELD_DEF(uint64_t, colsPerCoreDG);
TILING_DATA_FIELD_DEF(uint64_t, cols);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint64_t, colsPerUBDG);
TILING_DATA_FIELD_DEF(uint64_t, coreUbBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, tailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreTailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, colsLastCoreDG);
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupNormGrad_500, GroupNormGradEmptyTilingData)

struct GroupNormGradCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
    uint32_t vectorLen = 0;
    uint32_t blockSize = 0;
    bool isRegBase{false};
};

struct TilingCalculationParameters {
    uint32_t tilingKey = -1;
    uint64_t n = 0;
    uint64_t c = 0;
    uint64_t hxw = 0;
    uint64_t g = 0;
    uint64_t nxg = 0;
    uint64_t channelPerGroup = 0;
    uint32_t taskNumPerCore = 0;
    uint32_t taskNumPerTailCore = 0;
    uint32_t tailCore = 0;
    uint32_t mode0UbCapGNum = 0;
    uint32_t mode1UbCapCNum = 0;
    uint32_t mode1UbIterCNum = 0;
    uint32_t mode1UbTailCNum = 0;
    uint32_t mode2UbCapacityEle = 0;
    uint32_t mode2UbIterationNum = 0;
    uint32_t mode2UbTailNum = 0;
    uint64_t workSpaceSize = 0;
    uint32_t stage2CoreUsed = 0;
    uint32_t castEleNum = 0;
    uint32_t tailCastNum = 0;
    uint32_t coreBatchParts = 0;
    uint32_t coreBatchPartsTailRepeat = 0;
    uint64_t repeatTime4Stage2 = 0;
    uint32_t coreNumUsed = 0;
    bool dxIsRequire = true;
    bool dgammaIsRequire = true;
    bool dbetaIsRequire = true;
};

// TilingKey 的 dtype 组合后缀，供各 regbase tiling 模板复用
enum class GNGDtypeKey : int { FLOAT_FLOAT = 1, HALF_HALF = 2, BF16_BF16 = 3, HALF_FLOAT = 4, BF16_FLOAT = 5 };

// regbase 各 tiling 模板共享的公共参数，由基类统一解析填充
struct ParamsGroupNormGrad {
    uint32_t coreNum = 0;                     // AIV 核数
    uint64_t ubSizePlatForm = 0;              // 平台 UB 大小
    uint32_t vectorLen = 0;                   // 矢量寄存器长度(fp32 元素数)
    uint32_t blockSize = 0;                   // UB 搬运对齐块大小
    uint32_t sysWorkspaceSize = 0;            // 系统 workspace 大小
    int64_t N = 0;                            // x/dy 的 N 维
    int64_t C = 0;                            // x/dy 的 C 维
    int64_t G = 0;                            // num_groups
    int64_t CPerG = 0;                        // C / G
    int64_t HxW = 1;                          // 空间维乘积
    ge::DataType tTypeStr = ge::DT_UNDEFINED; // T type: x, dy, dx
    ge::DataType uTypeStr = ge::DT_UNDEFINED; // U type: mean, rstd, gamma, dgamma, dbeta
    bool isRegBase = false;
    bool dxIsRequire = true;
    bool dgammaIsRequire = true;
    bool dbetaIsRequire = true;
};

// group_norm_grad regbase 模板 tiling 公共基类：
// 仅内置公共参数 commonParams 与方法默认实现（参考 layer_norm_grad_v3 的 TilingBase 形态），
// 不内置任何模板私有成员；各 regbase tiling 模板（empty/mix 及后续新增）继承本类，
// 由注册路由机制按优先级调度
class GroupNormGradTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GroupNormGradTilingBase(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~GroupNormGradTilingBase() override = default;
    ParamsGroupNormGrad commonParams;

protected:
    // 1、获取INPUT/OUTPUT/ATTR信息（公共骨架，子类按需 override）
    ge::graphStatus GetShapeAttrsInfo() override;
    // 2、获取平台信息（从 compileInfo 统一取值填充 commonParams）
    ge::graphStatus GetPlatformInfo() override;
    // 3、模板能力自判（默认恒 true，特殊条件场景由子类 override）
    bool IsCapable() override;
    // 4、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 6、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、计算Workspace大小
    ge::graphStatus GetWorkspaceSize() override;
    // 8、保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 公共输入校验（默认严格版：要求所有维度大于等于1；空tensor等宽松场景由子类 override）
    virtual ge::graphStatus InputCheck(gert::Shape& dyShape);
    // 公共参数校验（默认严格版；C为0等宽松场景由子类 override）
    virtual ge::graphStatus ParamsCheck();
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_GROUPNORMGRAD_H
