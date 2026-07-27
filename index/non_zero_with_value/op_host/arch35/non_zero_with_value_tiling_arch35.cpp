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
 * \file non_zero_with_value_tiling_arch35.cpp
 * \brief NonZeroWithValue arch35 tiling —— 2D/fp32/transpose=true(坐标主序)专用。
 *        参照 index/non_zero 的 regbase tiling,简化为 2D 单路径 + value 通道。
 */
#include "non_zero_with_value_tiling_arch35.h"
#include <cmath>
#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

namespace optiling {
namespace {
constexpr size_t ATTR_TRANSPOSE_IDX = 0;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t DIM_ROW = 0;
constexpr size_t DIM_COL = 1;
constexpr size_t INPUT_RANK_2D = 2;
constexpr int64_t DIV_NUM = 2;
constexpr int64_t VALUE_CH_NUM = 3;           // idxOutBuf_ 的平面份数(压缩 idx + row + col)
constexpr int64_t NZV_DB_BUFFER_HOST = 2;     // 对齐 kernel base.h 的 inQueX_ double buffer
constexpr int64_t NZV_ADD_UB_BYTES = 72 * 32; // 对齐 kernel base.h 的 NZV_ADD_UB_SIZE
constexpr int64_t COORD_NUM = 2;              // 每个非零 2 坐标
constexpr int64_t QUICK_DIV_NUM_32 = 32;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t TMP_UB_SIZE = 2304L;         // 预留临时 UB(对齐 non_zero TMP_UB_SIZE_BIG)
constexpr int64_t TMP_UB_SIZE_BIGMASK = 9216L; // big_mask 预留
constexpr int64_t NUM_64 = 64;
constexpr int64_t WORKSPACE_SIZE_OFFSET = 9216L;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024 + WORKSPACE_SIZE_OFFSET;
constexpr int64_t WORKSPACE_SIZE_NULL = 16 * 1024 * 1024;

// TilingKey(方案1:单一通用路径。general 重算 mask + 按 tile 循环,覆盖单核小张量与超大张量;
// 因 value 通道输出遍必重载 x,重算 mask 近乎免费,full_load/big_mask 无收益,故不分路,见 02 §3)
constexpr int64_t TILING_KEY_NULL = 1000;
constexpr int64_t TILING_KEY_GENERAL = 1020;
} // namespace

class NonZeroWithValueTiling {
public:
    explicit NonZeroWithValueTiling(gert::TilingContext* context) : context_(context) {}

    ge::graphStatus Init(const NonZeroWithValueCompileInfo* compileInfo)
    {
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
        vRegSize_ = compileInfo->vRegSize;
        OP_CHECK_IF(coreNum_ <= 0 || ubSize_ <= 0 || vRegSize_ <= 0,
                    OP_LOGE(context_->GetNodeName(), "invalid compile info: coreNum/ubSize/vRegSize must be > 0"),
                    return ge::GRAPH_FAILED);

        auto attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        const bool* transposePtr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, transposePtr);
        // A5 仅支持 transpose=true(坐标主序);false 不支持(见 01 §6.2)
        OP_CHECK_IF(!(*transposePtr),
                    OP_LOGE(context_->GetNodeName(), "NonZeroWithValue A5 only supports transpose=true."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoTiling()
    {
        auto xStorage = context_->GetInputShape(INPUT_X_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
        const gert::Shape& xShape = xStorage->GetStorageShape();

        // x/value dtype 字节数(x 与 value 同 dtype):驱动 UB 记账与 VF 宽度按 dtype 缩放
        auto xDesc = context_->GetInputDesc(INPUT_X_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
        valueDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(xDesc->GetDataType()));
        OP_CHECK_IF(valueDtypeSize_ <= 0, OP_LOGE(context_->GetNodeName(), "invalid x dtype size."),
                    return ge::GRAPH_FAILED);

        // 严格 2D
        OP_CHECK_IF(xShape.GetDimNum() != INPUT_RANK_2D,
                    OP_LOGE(context_->GetNodeName(), "NonZeroWithValue requires 2D input."), return ge::GRAPH_FAILED);
        row_ = xShape.GetDim(DIM_ROW);
        col_ = xShape.GetDim(DIM_COL);
        // 整数乘法溢出守卫(红线规范):row*col 不得溢出 int64
        OP_CHECK_IF(row_ > 0 && col_ > 0 && col_ > (INT64_MAX / row_),
                    OP_LOGE(context_->GetNodeName(), "numel overflow: row * col exceeds int64 range."),
                    return ge::GRAPH_FAILED);
        numel_ = row_ * col_;

        if (numel_ == 0) {
            return DoTilingNull();
        }
        // int32 索引不变量:index/count 输出恒为 int32,线性 idx 与 count 上限均为 numel,
        // 故 numel 必须 ≤ INT32_MAX,否则 general.h 的 Arange(int32)截断、base.h 的 uint32 count 前缀和回绕。
        OP_CHECK_IF(numel_ > INT32_MAX,
                    OP_LOGE(context_->GetNodeName(), "numel exceeds int32 index capacity (INT32_MAX)."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(col_ <= 0, OP_LOGE(context_->GetNodeName(), "col must be > 0 for quick-div."),
                    return ge::GRAPH_FAILED);
        CalcColQuickDiv();

        // 核间切分:按展平 numel 线性切
        numPerCore_ = CeilDiv(numel_, coreNum_);
        realCoreNum_ = CeilDiv(numel_, numPerCore_);
        numTailCore_ = numel_ - (realCoreNum_ - 1) * numPerCore_;

        // tilingKey:非空一律走 general(单一通用路径,见上方常量注释 / 02 §3)
        tilingKey_ = TILING_KEY_GENERAL;

        CalcUbSplit();
        FillTilingData();

        context_->SetBlockDim(realCoreNum_);
        context_->SetTilingKey(tilingKey_);
        context_->SetScheduleMode(1);
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = WORKSPACE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

private:
    void CalcColQuickDiv()
    {
        // row = idx / col, col_idx = idx % col;quick-div 常量(对齐 non_zero CalcQuickDivParams)
        uint64_t c = static_cast<uint64_t>(col_);
        quickDivColK_ = static_cast<int64_t>(std::ceil(std::log2(static_cast<double>(c))));
        quickDivColM_ = static_cast<int64_t>(
            std::ceil(std::exp2(quickDivColK_ + QUICK_DIV_NUM_32) / static_cast<double>(c)) -
            std::exp2(QUICK_DIV_NUM_32));
    }

    void CalcUbSplit()
    {
        // UB 反推:kernel 实际同时驻留的 buffer(必须全部塞进 UB,否则 valOutBuf_ 基址越界 → VEC_ERROR):
        //   inQueX_(double buffer 2 份) = NZV_DB_BUFFER * 4
        //   idxOutBuf_(压缩 idx + row 平面 + col 平面 3 份) = 3 * 4
        //   valOutBuf_(压缩 value 1 份) = 4
        //   maskUb_ ≈ 0.5 字节/元素(CeilDiv(n,64)*32)
        // 合计 ≈ 24.5 字节/元素,取 BYTES_PER_ELEM=26 留裕量;固定 addUb_ + TMP 单列扣除。
        // 按 value dtype 尺寸记账:x(double buffer)+ valOut 用 valueDtypeSize_(=x/value dtype 字节数);
        // idx 三平面 + 掩码/临时恒 int32(index 输出恒 int32,与 x dtype 无关)。
        const int64_t bytesPerElem = (NZV_DB_BUFFER_HOST + 1) * valueDtypeSize_ + (VALUE_CH_NUM + 1) * INT32_BYTES;
        int64_t availUb = ubSize_ - TMP_UB_SIZE - NZV_ADD_UB_BYTES;
        int64_t vfElems = vRegSize_ / valueDtypeSize_;
        ubFactorNum_ = FloorDiv(availUb, bytesPerElem);
        // 对齐到 VF 寄存器宽度(vRegSize / value dtype 字节数,与 kernel vfLenV_ 一致)
        ubFactorNum_ = FloorDiv(ubFactorNum_, vfElems) * vfElems;
        if (ubFactorNum_ <= 0) {
            ubFactorNum_ = (vfElems > 0) ? vfElems : 1; // 除零守卫(G.EXP.22):fallback 可证非零
        }

        loopNumPerCore_ = numPerCore_ / ubFactorNum_;
        loopTailPerCore_ = numPerCore_ - loopNumPerCore_ * ubFactorNum_;
        loopNumTailCore_ = numTailCore_ / ubFactorNum_;
        loopTailTailCore_ = numTailCore_ - loopNumTailCore_ * ubFactorNum_;

        // 输出遍(与计数遍同 factor,坐标主序压缩)
        beforeNumO_ = (ubFactorNum_ / NUM_64) * NUM_64;
        if (beforeNumO_ <= 0) {
            beforeNumO_ = NUM_64;
        }
        loopNumO_ = numPerCore_ / beforeNumO_;
        loopTailO_ = numPerCore_ - loopNumO_ * beforeNumO_;
        loopNumTo_ = numTailCore_ / beforeNumO_;
        loopTailTo_ = numTailCore_ - loopNumTo_ * beforeNumO_;

        xInputSize_ = ubFactorNum_ * valueDtypeSize_;
        valueBufSize_ = ubFactorNum_ * valueDtypeSize_;
    }

    ge::graphStatus DoTilingNull()
    {
        tilingData_.set_row(row_);
        tilingData_.set_col(col_);
        tilingData_.set_numel(0);
        tilingData_.set_realCoreNum(1);
        tilingData_.set_tilingKey(TILING_KEY_NULL);
        SaveTiling();
        context_->SetBlockDim(1);
        context_->SetTilingKey(TILING_KEY_NULL);
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = WORKSPACE_SIZE_NULL;
        return ge::GRAPH_SUCCESS;
    }

    void FillTilingData()
    {
        tilingData_.set_row(row_);
        tilingData_.set_col(col_);
        tilingData_.set_numel(numel_);
        tilingData_.set_realCoreNum(realCoreNum_);
        tilingData_.set_numPerCore(numPerCore_);
        tilingData_.set_numTailCore(numTailCore_);
        tilingData_.set_ubFactorNum(ubFactorNum_);
        tilingData_.set_loopNumPerCore(loopNumPerCore_);
        tilingData_.set_loopTailPerCore(loopTailPerCore_);
        tilingData_.set_loopNumTailCore(loopNumTailCore_);
        tilingData_.set_loopTailTailCore(loopTailTailCore_);
        tilingData_.set_loopNumO(loopNumO_);
        tilingData_.set_beforeNumO(beforeNumO_);
        tilingData_.set_loopTailO(loopTailO_);
        tilingData_.set_loopNumTo(loopNumTo_);
        tilingData_.set_loopTailTo(loopTailTo_);
        tilingData_.set_xInputSize(xInputSize_);
        tilingData_.set_valueBufSize(valueBufSize_);
        tilingData_.set_quickDivColK(quickDivColK_);
        tilingData_.set_quickDivColM(quickDivColM_);
        tilingData_.set_tilingKey(tilingKey_);
        SaveTiling();
    }

    void SaveTiling()
    {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    }

    gert::TilingContext* context_ = nullptr;
    NonZeroWithValueTilingData tilingData_;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vRegSize_ = 0;
    int64_t valueDtypeSize_ = 4; // x/value dtype 字节数(DoTiling 按实际 x dtype 读),默认 fp32
    int64_t row_ = 0;
    int64_t col_ = 0;
    int64_t numel_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t numPerCore_ = 0;
    int64_t numTailCore_ = 0;
    int64_t ubFactorNum_ = 0;
    int64_t loopNumPerCore_ = 0;
    int64_t loopTailPerCore_ = 0;
    int64_t loopNumTailCore_ = 0;
    int64_t loopTailTailCore_ = 0;
    int64_t loopNumO_ = 0;
    int64_t beforeNumO_ = 0;
    int64_t loopTailO_ = 0;
    int64_t loopNumTo_ = 0;
    int64_t loopTailTo_ = 0;
    int64_t xInputSize_ = 0;
    int64_t valueBufSize_ = 0;
    int64_t quickDivColK_ = 0;
    int64_t quickDivColM_ = 0;
    int64_t tilingKey_ = 0;
};

static ge::graphStatus Tiling4NonZeroWithValue(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const NonZeroWithValueCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    NonZeroWithValueTiling tiling(context);
    OP_CHECK_IF(tiling.Init(compileInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Tiling init failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.DoTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "DoTiling failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4NonZeroWithValue(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NonZeroWithValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->vRegSize = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF(compileInfo->coreNum <= 0 || compileInfo->ubSize <= 0 || compileInfo->vRegSize <= 0,
                OP_LOGE(context->GetNodeName(), "invalid platform info."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NonZeroWithValue)
    .Tiling(Tiling4NonZeroWithValue)
    .TilingParse<NonZeroWithValueCompileInfo>(TilingPrepare4NonZeroWithValue);

} // namespace optiling
