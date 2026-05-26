/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_tiling.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_EXT_TILING_BASE_H_
#define SOFTMAX_GRAD_EXT_TILING_BASE_H_
#include <cmath>
#include "op_api/runtime2_util.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_util.h"

namespace optiling
{
using namespace Ops::NN::Optiling;
// ar小尾轴
BEGIN_TILING_DATA_DEF(SoftmaxGradExtARSmallRTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);              // x输入行数，A轴大小
TILING_DATA_FIELD_DEF(int64_t, r);              // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubFactor);       // ub内单次计算允许运算的最大行数（切分A）
TILING_DATA_FIELD_DEF(int64_t, aPerHeadCore);   // 大核处理的A轴行数
TILING_DATA_FIELD_DEF(int64_t, aPerTailCore);   // 小核处理的A轴行数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);   // 总共使用的核数
TILING_DATA_FIELD_DEF(int64_t, x2IsScalar);     // x2是否为单点输入
END_TILING_DATA_DEF;

// ar全载
BEGIN_TILING_DATA_DEF(SoftmaxGradExtARTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);             // x输入行数，A轴大小
TILING_DATA_FIELD_DEF(int64_t, r);             // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, rAligned);      // x输入列数，对齐后R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubFactor);      // UB内一次循环处理的a_in_in
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);  // 单核处理的行数a_in
TILING_DATA_FIELD_DEF(int64_t, rLoopCount);    // r / VL_Len
TILING_DATA_FIELD_DEF(int64_t, x2IsScalar);     // x2是否为单点输入
END_TILING_DATA_DEF;

// ar重计算
BEGIN_TILING_DATA_DEF(SoftmaxGradExtARRecomputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);               // x输入行数，A轴大小
TILING_DATA_FIELD_DEF(int64_t, r);               // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubFactor);        // UB处理的r_in
TILING_DATA_FIELD_DEF(int64_t, ubFactorTail);    // UB处理的r_in的尾块，值可能为0
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);    // 每个AIV处理的行数a_in
TILING_DATA_FIELD_DEF(int64_t, aLoopCountCeil);  // CeilDiv(r, r_in)
TILING_DATA_FIELD_DEF(int64_t, basicBlockLoop);  // 二分累加：循环次数，折叠点左半部分的block数量
TILING_DATA_FIELD_DEF(int64_t, mainFoldCount);   // 二分累加：折叠的块数，折叠点右半部分的block数量减1
TILING_DATA_FIELD_DEF(int64_t, x2IsScalar);     // x2是否为单点输入
// 二分累加
TILING_DATA_FIELD_DEF(int64_t, binAddRFactor);
TILING_DATA_FIELD_DEF(int64_t, binAddRLoop);
TILING_DATA_FIELD_DEF(int64_t, binAddRTotalLoop);
TILING_DATA_FIELD_DEF(int64_t, binAddRTail);
TILING_DATA_FIELD_DEF(int64_t, binAddBasicBlockLoop);
TILING_DATA_FIELD_DEF(int64_t, binAddMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, binAddCacheBufferCount);
TILING_DATA_FIELD_DEF(int64_t, binAddResultCacheID);
END_TILING_DATA_DEF;


constexpr int32_t TEMPLATE_AR_SMALL_R_PRIORITY = 50;
constexpr int32_t TEMPLATE_AR_FULL_LOAD_PRIORITY = 100;
constexpr int32_t TEMPLATE_AR_RECOMPUTE_PRIORITY = 200;


constexpr int64_t TILINGKEY_AR_SMALL_R = 500;
constexpr int64_t TILINGKEY_AR = 1000;
constexpr int64_t TILINGKEY_AR_RECOMPUTE = 2000;

// softmax_grad_ext
REGISTER_TILING_DATA_CLASS(SoftmaxGradExt, SoftmaxGradExtARSmallRTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxGradExt_1000, SoftmaxGradExtARTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxGradExt_2000, SoftmaxGradExtARRecomputeTilingData);


struct SoftmaxGradExtCompileInfo {
    // std::shared_ptr<AutoTilingCompileInfo> dslCompileInfo;
    bool isAscendC{false};
    int32_t coreNum{0};
    int64_t ubSize{0};
    int64_t blockSize{0};
    int64_t vlFp32{0};
    int64_t vlFp16{0};
};

constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;

constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t MAX_DIMS = 6;
constexpr int64_t DATA_BLOCK_COUNT = 16;

// binary add
constexpr int64_t CONST_ZERO = 0;
constexpr int64_t CONST_ONE = 1;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_THREE = 3;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t CONST_FIVE = 5;
constexpr int64_t CONST_SIX = 6;
constexpr int64_t CONST_SEVEN = 7;
constexpr int64_t CONST_EIGHT = 8;
constexpr int64_t CONST_SIXTY_THREE = 63;
constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;
constexpr uint64_t ULONG_BIT_LEN = 64;

constexpr uint16_t ROW_ZERO = 0;
constexpr uint16_t ROW_ONE = 1;
constexpr uint16_t ROW_TWO = 2;
constexpr uint16_t ROW_THREE = 3;
constexpr uint16_t ROW_FOUR = 4;
constexpr uint16_t ROW_FIVE = 5;
constexpr uint16_t ROW_SIX = 6;
constexpr uint16_t ROW_SEVEN = 7;

constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t ROW_FIVE_OFFSET = 5;
constexpr uint32_t ROW_SIX_OFFSET = 6;
constexpr uint32_t ROW_SEVEN_OFFSET = 7;

// 框架侧占位可以只预留32B（ttk正常），debugTool执行时需要预留16M
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;

class SoftmaxGradExtTilingBase : virtual public TilingBaseClass
{
public:
    explicit SoftmaxGradExtTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~SoftmaxGradExtTilingBase() override = default;

protected:
    bool IsCapable() override
    {
        return false;
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
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        return 0;
    }
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        // 计算workspace大小
        workspaceSize_ = MINIMAL_WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckFormatValid();
    virtual ge::graphStatus GetAndCheckDtypes();
    ge::graphStatus GetDimsAndCheckShapeValid();
    ge::graphStatus GetAndCheckAxes();

    std::string VectorToString(const std::vector<int64_t>& s);
    std::string VectorToString(const int64_t* s, int64_t size);

    int64_t usedCoreNums_{0};
    int64_t blockSize_{0};
    int64_t vlFp32_{0};
    int64_t vlFp16_{0};
    bool isX2Scalar_ = true;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT};
    ge::DataType yDtype_{ge::DataType::DT_FLOAT};
    int64_t xDtypeSize_{0};
    int64_t yDtypeSize_{0};

    int64_t xShapeSize_{0};
    int64_t xShapeSize2_{0};
    std::vector<int64_t> xShape_;

    int64_t a1_{DIM_NUM_ONE};
    int64_t r_{DIM_NUM_ONE};
    int64_t a0_{DIM_NUM_ONE};

    int64_t reduceAxes_{0};
};

// ar小尾轴
class SoftmaxGradExtTilingARSmallR : virtual public SoftmaxGradExtTilingBase
{
public:
    explicit SoftmaxGradExtTilingARSmallR(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradExtTilingBase(context)
    {
    }
    ~SoftmaxGradExtTilingARSmallR() override = default;

    void Reset(gert::TilingContext* context) override
    {
        SoftmaxGradExtTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

protected:
    SoftmaxGradExtARSmallRTilingData tilingData_;
};

// ar全载
class SoftmaxGradExtTilingAR : virtual public SoftmaxGradExtTilingBase
{
public:
    explicit SoftmaxGradExtTilingAR(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradExtTilingBase(context)
    {
    }
    ~SoftmaxGradExtTilingAR() override = default;

    void Reset(gert::TilingContext* context) override
    {
        SoftmaxGradExtTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

protected:
    SoftmaxGradExtARTilingData tilingData_;
};

// ar重计算
class SoftmaxGradExtTilingARRecompute : virtual public SoftmaxGradExtTilingBase
{
public:
    explicit SoftmaxGradExtTilingARRecompute(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradExtTilingBase(context)
    {
    }
    ~SoftmaxGradExtTilingARRecompute() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus BinarySummationTiling();
    int64_t Lcm(const int64_t a, const int64_t b);
    int64_t FindNearestPower2(const int64_t value);

private:
    SoftmaxGradExtARRecomputeTilingData tilingData_;

    int64_t ubFlexible_{0};
    int64_t baseFactor_{0};
    int64_t aLoopCountCeil_{0};
    int64_t aLoopCountFloor_{0};
    int64_t binAddRFactor_{0};
    int64_t binAddRLoop_{0};
    int64_t binAddRTotalLoop_{0};
    int64_t binAddRTail_{0};
    int64_t binAddBasicBlockLoop_{0};
    int64_t mainFoldCount_{0};
    int64_t binAddCacheBufferCount_{0};
    int64_t binAddResultCacheID_{0};
}; 

extern ge::graphStatus TilingForSoftmaxGradExt(gert::TilingContext* context);
extern ge::graphStatus TilingPrepareForSoftmaxGradExt(gert::TilingParseContext* context);

}  // namespace optiling

#endif  // SOFTMAX_GRAD_EXT_TILING_BASE_H_