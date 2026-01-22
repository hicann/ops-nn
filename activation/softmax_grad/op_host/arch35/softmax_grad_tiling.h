/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_tiling.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_TILING_BASE_H_
#define SOFTMAX_GRAD_TILING_BASE_H_

#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"


namespace optiling
{
// ar小尾轴
BEGIN_TILING_DATA_DEF(SoftmaxGradARSmallRTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalA0Len);
TILING_DATA_FIELD_DEF(int64_t, totalRLen);
TILING_DATA_FIELD_DEF(int64_t, tileA0Outer);
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, tileA0Len);
TILING_DATA_FIELD_DEF(int64_t, tileA0Tail);
TILING_DATA_FIELD_DEF(int64_t, rTileBase);
TILING_DATA_FIELD_DEF(int64_t, rAligned);
END_TILING_DATA_DEF;

// ar全载
BEGIN_TILING_DATA_DEF(SoftmaxGradARTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);             // x输入行数，A轴大小
TILING_DATA_FIELD_DEF(int64_t, r);             // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, rAligned);      // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubFactor);      // UB内一次循环处理的a_in_in
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);  // 单核处理的行数a_in
TILING_DATA_FIELD_DEF(int64_t, rLoopCount);    // r / VL_Len
END_TILING_DATA_DEF;

// ar重计算
BEGIN_TILING_DATA_DEF(SoftmaxGradARRecomputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);               // x输入行数，A轴大小
TILING_DATA_FIELD_DEF(int64_t, r);               // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubFactor);        // UB处理的r_in
TILING_DATA_FIELD_DEF(int64_t, ubFactorTail);    // UB处理的r_in的尾块，值可能为0
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);    // 每个AIV处理的行数a_in
TILING_DATA_FIELD_DEF(int64_t, aLoopCountCeil);  // CeilDiv(r, r_in)
TILING_DATA_FIELD_DEF(int64_t, basicBlockLoop);  // 二分累加：循环次数，折叠点左半部分的block数量
TILING_DATA_FIELD_DEF(int64_t, mainFoldCount);   // 二分累加：折叠的块数，折叠点右半部分的block数量减1
END_TILING_DATA_DEF;

// ara 全载
BEGIN_TILING_DATA_DEF(SoftmaxGradARATilingData)
TILING_DATA_FIELD_DEF(int64_t, totalRLen);
TILING_DATA_FIELD_DEF(int64_t, totalA0Len);
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, tileA0Outer);
TILING_DATA_FIELD_DEF(int64_t, tileA0Len);
TILING_DATA_FIELD_DEF(int64_t, tileA0Tail);
TILING_DATA_FIELD_DEF(int64_t, a1Outer);
TILING_DATA_FIELD_DEF(int64_t, a1Inner);
TILING_DATA_FIELD_DEF(int64_t, a1Tail);
END_TILING_DATA_DEF;

// ara重计算
BEGIN_TILING_DATA_DEF(SoftmaxGradARARecomputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalRLen);
TILING_DATA_FIELD_DEF(int64_t, totalA0Len);
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, tileA0Outer);
TILING_DATA_FIELD_DEF(int64_t, tileA0Len);
TILING_DATA_FIELD_DEF(int64_t, tileA0Tail);
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
constexpr int32_t TEMPLATE_ARA_FULL_LOAD_PRIORITY = 300;
constexpr int32_t TEMPLATE_ARA_RECOMPUTE_PRIORITY = 400;

constexpr int64_t TILINGKEY_AR_SMALL_R = 500;
constexpr int64_t TILINGKEY_AR = 1000;
constexpr int64_t TILINGKEY_AR_RECOMPUTE = 2000;
constexpr int64_t TILINGKEY_ARA = 10000;
constexpr int64_t TILINGKEY_ARA_RECOMPUTE = 20000;

// softmax_grad
REGISTER_TILING_DATA_CLASS(SoftmaxGrad_500, SoftmaxGradARSmallRTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxGrad_1000, SoftmaxGradARTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxGrad_2000, SoftmaxGradARRecomputeTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxGrad_10000, SoftmaxGradARATilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxGrad, SoftmaxGradARARecomputeTilingData);

// log_softmax_grad
REGISTER_TILING_DATA_CLASS(LogSoftmaxGrad_500, SoftmaxGradARSmallRTilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxGrad_1000, SoftmaxGradARTilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxGrad_2000, SoftmaxGradARRecomputeTilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxGrad_10000, SoftmaxGradARATilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxGrad, SoftmaxGradARARecomputeTilingData);

// confusion_softmax_grad
REGISTER_TILING_DATA_CLASS(ConfusionSoftmaxGrad_500, SoftmaxGradARSmallRTilingData);
REGISTER_TILING_DATA_CLASS(ConfusionSoftmaxGrad_1000, SoftmaxGradARTilingData);
REGISTER_TILING_DATA_CLASS(ConfusionSoftmaxGrad, SoftmaxGradARRecomputeTilingData);

struct SoftmaxGradCompileInfo {
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
constexpr int64_t MAX_DIMS = 8;
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

class SoftmaxGradTilingBase : virtual public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit SoftmaxGradTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~SoftmaxGradTilingBase() override = default;

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

    ge::DataType xDtype_{ge::DataType::DT_FLOAT};
    ge::DataType yDtype_{ge::DataType::DT_FLOAT};
    int64_t xDtypeSize_{0};
    int64_t yDtypeSize_{0};

    int64_t xShapeSize_{0};
    std::vector<int64_t> xShape_;

    int64_t a1_{DIM_NUM_ONE};
    int64_t r_{DIM_NUM_ONE};
    int64_t a0_{DIM_NUM_ONE};

    int64_t reduceAxes_{0};
};

// ar小尾轴
class SoftmaxGradTilingARSmallR : virtual public SoftmaxGradTilingBase
{
public:
    explicit SoftmaxGradTilingARSmallR(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~SoftmaxGradTilingARSmallR() override = default;

    void Reset(gert::TilingContext* context) override
    {
        SoftmaxGradTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

protected:
    SoftmaxGradARSmallRTilingData tilingData_;
};

// ar全载
class SoftmaxGradTilingAR : virtual public SoftmaxGradTilingBase
{
public:
    explicit SoftmaxGradTilingAR(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~SoftmaxGradTilingAR() override = default;

    void Reset(gert::TilingContext* context) override
    {
        SoftmaxGradTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

protected:
    SoftmaxGradARTilingData tilingData_;
};

// ar重计算
class SoftmaxGradTilingARRecompute : virtual public SoftmaxGradTilingBase
{
public:
    explicit SoftmaxGradTilingARRecompute(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~SoftmaxGradTilingARRecompute() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus BinarySummationTiling();
    int64_t Lcm(const int64_t a, const int64_t b);
    int64_t FindNearestPower2(const int64_t value);

private:
    SoftmaxGradARRecomputeTilingData tilingData_;

    int64_t ubFlexible_{0};
    int64_t baseFactor_{0};
    int64_t aLoopCountCeil_{0};
    int64_t aLoopCountFloor_{0};
};

// ara 全载
class SoftmaxGradARATiling : virtual public SoftmaxGradTilingBase
{
public:
    explicit SoftmaxGradARATiling(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~SoftmaxGradARATiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus ComputeBinaryAddParams();
    int64_t GetCacheID(const int64_t idx);

private:
    SoftmaxGradARATilingData tilingData_;
};

// ara重计算
class SoftmaxGradARARecomputeTiling : virtual public SoftmaxGradTilingBase
{
public:
    explicit SoftmaxGradARARecomputeTiling(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~SoftmaxGradARARecomputeTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus ComputeBinaryAddParams();
    int64_t GetCacheID(const int64_t idx);

private:
    SoftmaxGradARARecomputeTilingData tilingData_;

    // tiling 切分
    int64_t totalTiles_{0};
    int64_t tilesPerCore_{0};
    int64_t a0TileBase_{0};
    int64_t a0Outer_{0};
    int64_t tileA0Len_{0};
    int64_t tileA0Tail_{0};

    // 二分累加
    int64_t binAddRFactor_{0};
    int64_t binAddRLoop_{0};
    int64_t binAddRTotalLoop_{0};
    int64_t binAddRTail_{0};
    int64_t binAddBasicBlockLoop_{0};
    int64_t mainFoldCount_{0};
    int64_t binAddCacheBufferCount_{0};
    int64_t binAddResultCacheID_{0};
};

extern ge::graphStatus TilingForSoftmaxGrad(gert::TilingContext* context);
extern ge::graphStatus TilingPrepareForSoftmaxGrad(gert::TilingParseContext* context);

}  // namespace optiling

#endif  // SOFTMAX_GRAD_TILING_BASE_H_