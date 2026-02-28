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
 * \file softmax_v2_tiling.h
 * \brief
 */

#ifndef SOFTMAX_V2_TILING_BASE_H_
#define SOFTMAX_V2_TILING_BASE_H_
#include <cmath>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"
#include <vector>
#include <exe_graph/runtime/tiling_context.h>

using namespace Ops::NN::Optiling;
using namespace std;
namespace optiling
{
// ar小尾轴
BEGIN_TILING_DATA_DEF(SoftmaxV2ArSmallRTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalA0Len);     // A轴大小
TILING_DATA_FIELD_DEF(int64_t, totalRLen);      // R轴大小
TILING_DATA_FIELD_DEF(int64_t, totalTiles);     // tiling块数量
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);   // 单核处理的tiling块数
TILING_DATA_FIELD_DEF(int64_t, tileA0Len);      // tiling块在A方向的长度
TILING_DATA_FIELD_DEF(int64_t, tileA0Tail);     // tiling块在A方向的尾块长度
TILING_DATA_FIELD_DEF(int64_t, rTileBase);      // r基础块，用于TransDataTo5HD
TILING_DATA_FIELD_DEF(int64_t, rAligned);
END_TILING_DATA_DEF;

// ar全载
BEGIN_TILING_DATA_DEF(SoftmaxV2ARTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);             // x输入行数，A轴大小
TILING_DATA_FIELD_DEF(int64_t, r);             // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, rAligned);      // x输入列数，R轴大小
TILING_DATA_FIELD_DEF(int64_t, ubFactor);      // UB内一次循环处理的a_in_in
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);  // 单核处理的行数a_in
TILING_DATA_FIELD_DEF(int64_t, rLoopCount);    // r / VL_Len
END_TILING_DATA_DEF;

// ar重计算
BEGIN_TILING_DATA_DEF(SoftmaxV2ArRecomputeTilingData)
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
BEGIN_TILING_DATA_DEF(SoftmaxV2ARATilingData)
TILING_DATA_FIELD_DEF(int64_t, totalA1Len);
TILING_DATA_FIELD_DEF(int64_t, totalRLen);
TILING_DATA_FIELD_DEF(int64_t, totalA0Len);
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, a1Outer);
TILING_DATA_FIELD_DEF(int64_t, tileA1Len);
TILING_DATA_FIELD_DEF(int64_t, tileA1Tail);
TILING_DATA_FIELD_DEF(int64_t, a0Outer);
TILING_DATA_FIELD_DEF(int64_t, tileA0Len);
TILING_DATA_FIELD_DEF(int64_t, tileA0Tail);
// 二分累加
TILING_DATA_FIELD_DEF(uint16_t, binaryAddK);
TILING_DATA_FIELD_DEF(uint16_t, binaryAddLast);
TILING_DATA_FIELD_DEF(uint16_t, binaryAddInnerLoop);
TILING_DATA_FIELD_DEF(uint16_t, remainderLoopCount);
TILING_DATA_FIELD_DEF(uint16_t, quotientLoopCount);
TILING_DATA_FIELD_DEF(uint16_t, remainderTailCount);
TILING_DATA_FIELD_DEF(uint32_t, remainderOffset);
TILING_DATA_FIELD_DEF(uint32_t, baseLineOffset);
TILING_DATA_FIELD_DEF(uint32_t, validNumInXUb);
TILING_DATA_FIELD_DEF(uint32_t, quotientTailOffset);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset0);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset1);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset2);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset3);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset4);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset5);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset6);
TILING_DATA_FIELD_DEF(uint32_t, remainderTailOffset7);
END_TILING_DATA_DEF;

// ara重计算
BEGIN_TILING_DATA_DEF(SoftmaxV2ARARecomputeTilingData)
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

REGISTER_TILING_DATA_CLASS(SoftmaxV2_500, SoftmaxV2ArSmallRTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxV2_1000, SoftmaxV2ARTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxV2_2000, SoftmaxV2ArRecomputeTilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxV2_10000, SoftmaxV2ARATilingData);
REGISTER_TILING_DATA_CLASS(SoftmaxV2, SoftmaxV2ARARecomputeTilingData);

// log_softmax_v2
REGISTER_TILING_DATA_CLASS(LogSoftmaxV2_500, SoftmaxV2ArSmallRTilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxV2_1000, SoftmaxV2ARTilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxV2_2000, SoftmaxV2ArRecomputeTilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxV2_10000, SoftmaxV2ARATilingData);
REGISTER_TILING_DATA_CLASS(LogSoftmaxV2, SoftmaxV2ARARecomputeTilingData);

struct SoftmaxV2CompileInfo {
    int32_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    int64_t vlFp32;
    int64_t vlFp16;
};

constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t DATA_BLOCK_COUNT = 16;
constexpr int64_t SMALL_A_LEN = 160;

constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t MAX_DIMS = 8;

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
constexpr int64_t CONST_SIXTEEN = 16;
constexpr int64_t CONST_SIXTY_THREE = 63;
constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;
const uint64_t ULONG_BIT_LEN = 64;

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

class SoftmaxV2TilingBase : virtual public TilingBaseClass
{
public:
    explicit SoftmaxV2TilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~SoftmaxV2TilingBase() override = default;

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

    int64_t usedCoreNums_;
    int64_t blockSize_;
    int64_t vlFp32_;
    int64_t vlFp16_;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT};
    ge::DataType yDtype_{ge::DataType::DT_FLOAT};
    int64_t xDtypeSize_{0};
    int64_t yDtypeSize_{0};

    int64_t xShapeSize_;
    vector<int64_t> xShape_;

    int64_t a1_{DIM_NUM_ONE};
    int64_t r_{DIM_NUM_ONE};
    int64_t a0_{DIM_NUM_ONE};

    int64_t reduceAxes_;
    bool halfToFloat_;
};

// ar小尾轴
class SoftmaxV2TilingArSmallR : virtual public SoftmaxV2TilingBase
{
public:
    explicit SoftmaxV2TilingArSmallR(gert::TilingContext* context) : TilingBaseClass(context), SoftmaxV2TilingBase(context)
    {
    }
    ~SoftmaxV2TilingArSmallR() override = default;

    void Reset(gert::TilingContext* context) override
    {
        SoftmaxV2TilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

protected:
    SoftmaxV2ArSmallRTilingData tilingData_;
};

// ar全载
class SoftmaxV2TilingAR : virtual public SoftmaxV2TilingBase
{
public:
    explicit SoftmaxV2TilingAR(gert::TilingContext* context) : TilingBaseClass(context), SoftmaxV2TilingBase(context)
    {
    }
    ~SoftmaxV2TilingAR() override = default;

    void Reset(gert::TilingContext* context) override
    {
        SoftmaxV2TilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

protected:
    SoftmaxV2ARTilingData tilingData_;
};

// ar重计算
class SoftmaxV2TilingARRecompute : virtual public SoftmaxV2TilingBase
{
public:
    explicit SoftmaxV2TilingARRecompute(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxV2TilingBase(context)
    {
    }
    ~SoftmaxV2TilingARRecompute() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus BinarySummationTiling();
    int64_t Lcm(const int64_t a, const int64_t b);
    int64_t FindNearestPower2(const int64_t value);
    int64_t Gcd(int64_t a, int64_t b);

private:
    SoftmaxV2ArRecomputeTilingData tilingData_;

    int64_t ubFlexible_ = 0;
    int64_t baseFactor_ = 0;
    int64_t aLoopCountCeil_ = 0;
    int64_t aLoopCountFloor_ = 0;
};

// ara 全载
class SoftmaxV2ARATiling : virtual public SoftmaxV2TilingBase
{
public:
    explicit SoftmaxV2ARATiling(gert::TilingContext* context) : TilingBaseClass(context), SoftmaxV2TilingBase(context)
    {
    }
    ~SoftmaxV2ARATiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus BinaryAddTiling();

private:
    SoftmaxV2ARATilingData tilingData_;

    int64_t a0TileBase_;
    int64_t tileA0Len_;
    int64_t totalTiles_;
    int64_t tilesPerCore_;

    int64_t binaryAddQuotient_{0};
};

// ara重计算
class SoftmaxV2ARARecomputeTiling : virtual public SoftmaxV2TilingBase
{
public:
    explicit SoftmaxV2ARARecomputeTiling(gert::TilingContext* context)
        : TilingBaseClass(context), SoftmaxV2TilingBase(context)
    {
    }
    ~SoftmaxV2ARARecomputeTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus ComputeBinaryAddParams();
    int64_t GetCacheID(const int64_t idx);

private:
    SoftmaxV2ARARecomputeTilingData tilingData_;

    // tiling 切分
    int64_t totalTiles_;
    int64_t tilesPerCore_;
    int64_t a0TileBase_;
    int64_t a0Outer_;
    int64_t tileA0Len_;
    int64_t tileA0Tail_;

    // 二分累加
    int64_t binAddRFactor_;
    int64_t binAddRLoop_;
    int64_t binAddRTotalLoop_;
    int64_t binAddRTail_;
    int64_t binAddBasicBlockLoop_;
    int64_t mainFoldCount_;
    int64_t binAddCacheBufferCount_ = 1;
    int64_t binAddResultCacheID_ = 0;
};

extern ge::graphStatus TilingForSoftmaxV2(gert::TilingContext* context);
extern ge::graphStatus TilingPrepareForSoftmaxV2(gert::TilingParseContext* context);

}  // namespace optiling

#endif  // SOFTMAX_V2_TILING_BASE_H_