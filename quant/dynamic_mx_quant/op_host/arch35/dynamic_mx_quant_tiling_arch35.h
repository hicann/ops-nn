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
 * \file dynamic_mx_quant_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_H
#include <cstdint>
#include <vector>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
using namespace Ops::NN::Optiling;

BEGIN_TILING_DATA_DEF(DynamicMxQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);        // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, blockFactor);        // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);    // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, ubDim);              // 合轴后，ubfactor所切的轴
TILING_DATA_FIELD_DEF(int64_t, uo);                 // 切分轴上的循环次数
TILING_DATA_FIELD_DEF(int64_t, ubFactor);           // 单次循环要处理的数据大小
TILING_DATA_FIELD_DEF(int64_t, tailUbFactor);       // 尾循环要处理的数据大小
TILING_DATA_FIELD_DEF(int64_t, roundMode);          // 数据类型转换的模式
TILING_DATA_FIELD_DEF(int64_t, dstType);            // 输出y的数据类型
TILING_DATA_FIELD_DEF(int64_t, blockSize);          // 进行微缩的数据块大小
TILING_DATA_FIELD_DEF(int64_t, scaleAlg);           // scale计算方法
TILING_DATA_FIELD_DEF(int64_t, blockSizeNumInAxis); // 在axis轴上有多少个blocksize
TILING_DATA_FIELD_DEF(int64_t, tailBlockSize);      // 指定轴要进行微缩的最后一个数据块大小
TILING_DATA_FIELD_DEF(int64_t, isPad);              // axis指定的轴是否需要补到blocksize的整数倍
TILING_DATA_FIELD_DEF(int64_t, isTailAxis);         // 是否为尾轴场景
TILING_DATA_FIELD_DEF(int64_t, preAxisSize);        // 合轴后axis前面轴的大小
TILING_DATA_FIELD_DEF(int64_t, postAxisSize);       // 合轴后axis后面轴的大小
TILING_DATA_FIELD_DEF(int64_t, mxScaleSize);        // scale数据大小
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, ubFactorDim0TailAxis);                  // 尾轴方向计算时dim0方向UB处理的行数
TILING_DATA_FIELD_DEF(int64_t, ubFactorDim1TailAxis);                  // 尾轴方向计算时dim1方向UB处理的block数
TILING_DATA_FIELD_DEF(int64_t, blockFactorDim0TailAxis);               // 尾轴方向计算时dim0方向正常核处理的循环次数
TILING_DATA_FIELD_DEF(int64_t, blockFactorDim1TailAxis);               // 尾轴方向计算时dim1方向正常核处理的循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactorDim0TailAxis);           // 尾轴方向计算时dim0方向尾核处理的循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockCountDim1TailAxis);            // 尾轴方向计算时dim1方向尾循环处理的block数
TILING_DATA_FIELD_DEF(int64_t, tailCoreStartIdxDim0TailAxis);          // dim0方向尾尾核开始索引
TILING_DATA_FIELD_DEF(int64_t, cutNumberForDim1TailAxis);              // 尾轴方向计算时1轴切分的次数
TILING_DATA_FIELD_DEF(int64_t, inputShapeDim1TailAxis);                // 尾轴大小
TILING_DATA_FIELD_DEF(int64_t, tailUbFactorDim0TailAxis);              // 尾轴行方向尾核最后一个循环处理的行数
TILING_DATA_FIELD_DEF(int64_t, isSplitDim1TailAxis);                   // 尾轴行方向是否切分1轴
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactorDim1ForSplitDim1);       // 尾轴方向切分1轴后尾核dim1方向循环次数
TILING_DATA_FIELD_DEF(int64_t, tailLoopBlockTailCoreDim1ForSplitDim1); // 尾轴方向切分1轴后尾核dim1方向尾循环block数
TILING_DATA_FIELD_DEF(int64_t, dim1EachCoreForSplitDim1);              // 尾轴方向切分1轴后正常核处理的BLOCK数
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(DynamicMxQuant4OptimizeTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);  // 总核数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);   // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, roundMode);     // 数据类型转换的模式
TILING_DATA_FIELD_DEF(int64_t, dstType);       // 输出y的数据类型
TILING_DATA_FIELD_DEF(int64_t, blockSize);     // 进行微缩的数据块大小
TILING_DATA_FIELD_DEF(int64_t, isPad);         // 量化轴最后一个block无法被blockSize整除时，为True
TILING_DATA_FIELD_DEF(int64_t, tailBlockSize); // 指定量化轴最后一个blockSize大小
TILING_DATA_FIELD_DEF(int64_t, scaleAlg);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, quantAxisSize);   // 优化非尾轴模板量化轴大小
TILING_DATA_FIELD_DEF(int64_t, preAxisSize);     // 合轴后axis前面轴的大小
TILING_DATA_FIELD_DEF(int64_t, postAxisSize);    // 合轴后axis后面轴的大小
TILING_DATA_FIELD_DEF(int64_t, mAlignSize);      // 量化轴对齐blockSize之后元素个数
TILING_DATA_FIELD_DEF(int64_t, nAlignSize);      // 融合尾轴对齐32,64,128之后元素个数
TILING_DATA_FIELD_DEF(int64_t, mAlignBlockSize); // 量化轴对齐blockSize之后block的个数，实际上等于blockSizeNumInAxis
TILING_DATA_FIELD_DEF(int64_t, nAlignBlockSize); // 融合尾轴对齐32,64,128之后block的个数
TILING_DATA_FIELD_DEF(int64_t, mAlignGroupSize); // 量化轴对齐blockSize*2（一个Group）之后Group的个数
TILING_DATA_FIELD_DEF(int64_t, quantAxisIsOdd);  // 量化轴是否是奇数，如果是奇数，则有些group会有一个全0的dummy block
TILING_DATA_FIELD_DEF(int64_t, totalGroupNum);   // 当前shape总共需要多少个Group才能计算完
TILING_DATA_FIELD_DEF(int64_t, groupPerCore);    // 每个核计算多少个Group
TILING_DATA_FIELD_DEF(int64_t, groupPerTail);    // 尾核计算多少个Group
TILING_DATA_FIELD_DEF(int64_t, groupPerUb);      // 每个UB可以放下多少个Group
TILING_DATA_FIELD_DEF(
    int64_t, totalBlockNum); // 总共处理的block数量，与输入block数可能不同，此处为对齐成group之后的block数量
TILING_DATA_FIELD_DEF(int64_t, blockNumPerTask); // 每个任务处理多少个blcok
TILING_DATA_FIELD_DEF(int64_t, totalTaskNum);    // 总任务数量，用总共处理的block数量除以每个任务处理多少个block
TILING_DATA_FIELD_DEF(int64_t, rowPerHeadCore);
TILING_DATA_FIELD_DEF(int64_t, rowPerTailCore);
TILING_DATA_FIELD_DEF(int64_t, needPadPostAxis); // 融合尾轴是否需要对齐
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicMxQuant, DynamicMxQuantTilingData)
REGISTER_TILING_DATA_CLASS(DynamicMxQuant_20000, DynamicMxQuant4OptimizeTilingData)
REGISTER_TILING_DATA_CLASS(DynamicMxQuant_10000, DynamicMxQuant4OptimizeTilingData)

struct DynamicMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicMxQuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    uint32_t workspaceSize{0};
    int64_t axis{0};
    int64_t roundMode;
    int64_t dstType{0};
    int64_t blockSize{0};
    int64_t scaleAlg{0};
    bool isTailAxis{false};
    bool isPad{false};
    int64_t tailBlockSize{0};
    int64_t blockSizeNumInAxis{0};
    int64_t preAxisSize{1};
    int64_t axisSize{1};
    int64_t quantAxisSize{1};
    int64_t postAxisSize{1};
    int64_t mxScaleSize{0};
    int64_t ubDim{0};
    int64_t ubFactor{0};
    int64_t uo{1};
    int64_t usedCoreNum{0};
    int64_t tailUbFactor{0};
    int64_t blockFactor{0};
    int64_t tailBlockFactor{0};
    int64_t tilingKey{0};
    int64_t ubFactorDim0TailAxis{0};
    int64_t ubFactorDim1TailAxis{0};
    int64_t blockFactorDim0TailAxis{0};
    int64_t blockFactorDim1TailAxis{0};
    int64_t tailBlockFactorDim0TailAxis{0};
    int64_t tailBlockCountDim1TailAxis{0};
    int64_t tailCoreStartIdxDim0TailAxis{0};
    int64_t cutNumberForDim1TailAxis{0};
    int64_t inputShapeDim1TailAxis{0};
    int64_t tailUbFactorDim0TailAxis{0};
    int64_t isSplitDim1TailAxis{0};
    int64_t tailBlockFactorDim1ForSplitDim1{0};
    int64_t tailLoopBlockTailCoreDim1ForSplitDim1{0};
    int64_t dim1EachCoreForSplitDim1{0};
    bool tailAxisOptimize{false};
    int64_t mAlignSize{1};
    int64_t nAlignSize{1};
    int64_t mAlignBlockSize{1};
    int64_t nAlignBlockSize{1};
    int64_t mAlignGroupSize{1};
    int64_t quantAxisIsOdd{0};
    int64_t totalGroupNum{0};
    int64_t groupPerCore{0};
    int64_t groupPerTail{0};
    int64_t groupPerUb{0};
    int64_t totalBlockNum{0};
    int64_t blockNumPerTask{0};
    int64_t totalTaskNum{0};
    int64_t rowPerHeadCore{0};
    int64_t rowPerTailCore{0};
    bool needPadPostAxis{false};
    bool isOptimize{false};
};

enum class RoundModeList
{
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

class DynamicMxQuantOptimzieTiling : public TilingBaseClass {
public:
    explicit DynamicMxQuantOptimzieTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~DynamicMxQuantOptimzieTiling() override
    {}

    const std::string nodeName = "DynamicMxQuantOptimzie";
    DynamicMxQuant4OptimizeTilingData tilingData;
    DynamicMxQuantTilingParam tilingParams;

    ge::graphStatus CheckDtype();
    ge::graphStatus GetAttr();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void SetTilingKey();
    void PrintTilingData();
    RoundModeList GetRoundMode(const std::string& roundMode);
    void CalScaleSize(const gert::Shape& inputShape);
    ge::graphStatus CalShapeAlign(int64_t& nAlignNum);

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->PostTiling->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_H