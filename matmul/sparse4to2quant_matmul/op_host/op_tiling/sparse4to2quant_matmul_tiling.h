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
 * \file sparse4to2quant_matmul_tiling.h
 * \brief
 */

#ifndef SPARSE_4TO2_QUANT_MATMUL_TILING_H
#define SPARSE_4TO2_QUANT_MATMUL_TILING_H

#include <string>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "../../op_kernel/sparse4to2quant_matmul_tiling_data.h"

namespace optiling {
struct BasicTiling {
    uint64_t usedCoreNum = 1;
    uint64_t singleCoreK = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t depthA1 = 1;
    uint64_t depthB1 = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint64_t iterateOrder = 0;
    uint64_t dbL0c = 1;
    uint64_t isMclash = 0;
    uint64_t isNclash = 0;
    uint64_t mTileCntl2 = 0;
    uint64_t nTileCntl2 = 0;
    uint64_t mTileBlock = 0;
    uint64_t nTileBlock = 0;
    uint64_t calOrder = 0;
};

template <typename T>
inline bool CheckNumberIsValid(const T& num, const std::string& opName, const std::string& description)
{
    if (num > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGW(
            opName.c_str(), "%s size is greater than INT32_MAX or less than 0, num:%s", description.c_str(),
            std::to_string(num).c_str());
        return true;
    }
    return false;
};

struct Sparse4to2QuantMatmulInfo {
    bool hasBias = false;
    uint64_t mSize = 0UL;
    uint64_t kaSize = 0UL;
    uint64_t kbSize = 0UL;
    uint64_t nSize = 0UL;
    ge::DataType aDtype = ge::DT_INT8;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_BF16;
    ge::DataType scaleDtype = ge::DT_FLOAT;
    ge::DataType perTokenScaleDtype = ge::DT_FLOAT;
    uint32_t xScaleDim = 0;
    uint32_t weightScaleDim = 0;
    int64_t outDtype = 0L;
    const char* opName = nullptr;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_FRACTAL_NZ;
    ge::Format cFormat = ge::FORMAT_ND; // 新增数据成员要修改Reset函数
};

struct Sparse4to2QuantMatmulCompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    bool supportL0c2Out;
    bool supportL12BtBf16;
    platform_ascendc::SocVersion socVersion;
    std::string socVersionStr = "";
};

class Sparse4to2QuantMatmulTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit Sparse4to2QuantMatmulTiling(gert::TilingContext* context) : TilingBaseClass(context){};
    ~Sparse4to2QuantMatmulTiling() override = default;

    BasicTiling basicTiling_;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    void PrintBasicTiling() const;
    ge::graphStatus CheckContext();
    bool AnalyzeDtype();
    bool AnalyzeInputs();
    bool DoBasicTiling();
    bool CalcL0Tiling();
    bool CalcL1Tiling();
    void DoL2CacheTiling();
    void SetMatmulTilingFromBasicTiling();
    bool InitTilingData(matmul_tiling::MultiCoreMatmulTiling& mm);
    bool GetUbDequantExtreSpace();
    ge::graphStatus CalcUbTiling(uint64_t& baseM, uint64_t& baseN);
    std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> CalcCoreDistribution(
        uint64_t mCnt, uint64_t nCnt, uint64_t calcOrder, uint64_t round, uint64_t usedCoreNum) const; // 计算核的排布
    bool SetBase(const std::vector<uint64_t>& mBases, const std::vector<uint64_t>& nBases);
    void CompareBase(std::vector<uint64_t>& basicMetrics, uint64_t baseM, uint64_t baseN);
    bool CheckCalcAndMemRatio(uint64_t baseM, uint64_t baseN) const;
    bool CheckL2Load(std::vector<uint64_t>& basicMetrics, uint64_t coreClash, uint64_t firstL2Load) const;
    bool CheckMTE1(uint64_t baseM, uint64_t baseN) const;
    bool CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c = 1) const;
    uint64_t GetMaxBaseN() const;
    bool GetBaseK(uint64_t baseM, uint64_t baseN);
    void CalcClashAndFirstL2Load(
        uint64_t& coreClash, uint64_t& firstL2Load, uint64_t mCnt, uint64_t nCnt, uint64_t round) const;
    bool CheckDbL0c() const;
    void InitBasicMetrics(std::vector<uint64_t>& basicMetrics);
    bool IsMNSmallForMultiCores(uint64_t coreNum) const;
    void ProcessMNSmallShape(uint64_t baseM, uint64_t baseN, uint64_t coreNum);
    uint64_t CalcL1LoadSize(uint64_t baseM, uint64_t baseN) const;
    int8_t CheckLoadAndCalcSize(
        uint64_t baseM, uint64_t baseN, uint64_t bestRound, uint64_t round, uint64_t& bestLoadSize) const;
    void ModifyBase(uint64_t& baseM, uint64_t& baseN) const;
    bool GetStepK(uint64_t& stepKa, uint64_t& stepKb) const;
    void CorrectStepK(uint64_t& bigStepK, uint64_t& smallStepK, uint64_t minStepK) const;
    uint64_t CalcL1SizeForBiasAndScale();
    uint64_t GetTotalCnt(uint64_t baseM, uint64_t baseN) const;
    bool IsTilingDataInvalid() const;
    void ResetBase(const uint64_t l0CSize);
    uint64_t GetTotalSize(uint64_t m, uint64_t ka, uint64_t kb, uint64_t n) const;
    void CalcTileCnt(
        uint64_t outOriShape, uint64_t innerOriShape, uint64_t outBase, uint64_t innerBase,
        std::vector<std::tuple<uint64_t, uint64_t>>& tileCnt) const;
    bool IsTileClash(
        uint64_t outSplit, uint64_t innerSplit, std::tuple<uint64_t, uint64_t>& tileClash,
        const std::tuple<uint64_t, uint64_t, uint64_t>& params) const;
    uint64_t GetCalcOrder(uint64_t mCnt, uint64_t nCnt, uint64_t mSize, uint64_t nSize, uint64_t usedCoreNum) const;
    bool CheckTileTail(uint64_t outTail, uint64_t innerTail, uint64_t outL2SplitTmp, uint64_t innerL2SplitTmp) const;
    bool CheckTileClash(
        const std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>& tileInfo,
        const std::tuple<uint64_t, uint64_t, uint64_t>& params,
        std::vector<std::tuple<uint64_t, uint64_t>>& tileClash) const;
    uint64_t CalcTile(
        uint64_t& outTile, uint64_t& innerTile, uint64_t& outL2Split, uint64_t& innerL2Split,
        const std::tuple<uint64_t, uint64_t, double>& params) const;
    void DivisibleCoreLayout(uint64_t mCnt, uint64_t nCnt, uint64_t& calcOrder, uint64_t round) const;
    void DetermineCalcOrder();
    void SetCalcOrderinMNClashCase(uint64_t mTotalCnt, uint64_t nTotalCnt);
    template <typename T>
    T GetShapeWithDataType(T size, ge::DataType dtype) const
    {
        if (dtype == ge::DT_INT4) {
            return size + size;
        } else {
            return size / static_cast<T>(ge::GetSizeByDataType(dtype));
        }
    }

private:
    bool isBf16Opt_ = false;
    bool isUbQuant_ = false;
    uint64_t workspaceSize_ = 0;
    Sparse4to2QuantMatmulCompileInfo compileInfo_;
    gert::TilingContext* context;
    platform_ascendc::SocVersion socVersion_;
    Sparse4to2QuantMatmulInfo inputParams_;
    SparseQmm::Sparse4to2QuantMatmulTilingData tilingData_;
};
} // namespace optiling

#endif // SPARSE_4TO2_QUANT_MATMUL_TILING_H