/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_EINSUM_FUSION_PASS_H
#define NN_EINSUM_FUSION_PASS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/es_tensor_holder.h"
#include "graph/graph.h"

namespace ops {

class __attribute__((visibility("default"))) EinsumPass : public ge::fusion::PatternFusionPass {
protected:
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;

    std::unique_ptr<ge::Graph> Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;

private:
    enum EinsumDimensionType { kBroadCast = 0, kBatch, kFree, kContract, kReduce, kDimTypeNum };

    struct LabelInfo {
        std::string labels;
        std::vector<int32_t> indices;
    };

    using LabelCount = std::map<char, uint32_t>;
    using DimensionType2LabelInfo = std::vector<LabelInfo>;

    // Handle* functions for specific equation patterns (dynamic shape)
    ge::fusion::GraphUniqPtr HandleDynamicABCxCDE2ABDE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                       const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxAECD2ACEB(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxADBE2ACBE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxCDE2ABE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCxCD2ABD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                              const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCxDC2ABD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                              const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCxABD2DC(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                              const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleDynamicABCxDEC2ABDE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                       const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleDynamicABCxABDE2DEC(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                       const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxAECD2ACBE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxACBE2AECD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxECD2ABE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleDynamicABCDxABE2ECD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                       const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxACBE2ADBE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxABDE2ABCE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxABCE2ABDE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxAEBD2AEBC(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxABCE2ACDE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCxABD2ACD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                               const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABxCB2AC(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                            const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCxACD2ABD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                               const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCxADC2ABD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                               const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxCED2ABCE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                 const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxEBCD2BCAE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxDABE2CABE(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleAxB2AB(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                          const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleABCDxAECD2EB(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr HandleDynamicABCDxEB2AECD(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                       const ge::GNode& matchedNode);

    // Generic decomposition paths
    ge::fusion::GraphUniqPtr SplitOpInFuzzScene(const std::string& equation,
                                                const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                const ge::GNode& matchedNode);
    ge::fusion::GraphUniqPtr SplitDynamicFuzzScene(const std::string& equation,
                                                   const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                   const ge::GNode& matchedNode);
    ge::GNode ResolveLastOutputNode();
    bool CreateDynamicInputHolders(ge::es::EsGraphBuilder& builder,
                                   const std::vector<ge::fusion::SubgraphInput>& subgraphInputs);

    // MeetRequirements helpers (split from the original oversized function)
    bool CheckEquationFormat(const ge::GNode& matchedNode, std::string& equation,
                             std::vector<std::string>& inEquations);
    bool CheckLabelConsistency(const ge::GNode& matchedNode, const std::string& equation,
                               const std::vector<std::string>& inEquations, std::vector<ge::DataType>& inputDtypes,
                               std::vector<std::vector<int64_t>>& inputDims);
    bool CheckBroadcastConsistency(const std::string& equation, const std::vector<std::string>& inEquations,
                                   const std::vector<ge::DataType>& inputDtypes,
                                   const std::vector<std::vector<int64_t>>& inputDims);

    // SplitDynamicFuzzScene helpers (split from the original oversized function)
    ge::Status PrepareDynamicLabels(const std::string& equation, const ge::GNode& matchedNode, size_t numOps,
                                    std::vector<std::vector<uint8_t>>& opLabels, int64_t& ellNumDim, int64_t& permIndex,
                                    int64_t& outNumDim, std::vector<int64_t>& dimCounts);
    ge::Status ReduceAndSumPairForDynamic(const ge::GNode& matchedNode, int64_t permIndex, int64_t outNumDim,
                                          std::vector<int64_t>& dimCounts, std::vector<int64_t>& sumDims);
    ge::Status ExecuteDynamicFuzz(const ge::GNode& matchedNode, size_t numOps, int64_t permIndex, int64_t outNumDim,
                                  std::vector<int64_t>& dimCounts);

    // Helper: build batchmatmul replacement graph
    ge::fusion::GraphUniqPtr BuildBatchMatmulReplacement(bool adjX1, bool adjX2,
                                                         const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                         const ge::GNode& matchedNode);

    // Equation parsing helpers
    void ParseEquation(const std::string& equation, std::vector<std::string>& inEquations, std::string& outEquation);
    void SplitStr2Vector(const std::string& input, const std::string& delimiter,
                         std::vector<std::string>& output) const;
    std::string FilterInvalidLabel(const std::string& equation) const;
    void CountLabels(const std::string& equation, LabelCount& labelCount, std::set<char>& labels) const;
    void MapDimensionType(const std::set<char>& labels, const LabelCount& input0LabelCount,
                          const LabelCount& input1LabelCount, const LabelCount& outputLabelCount);
    void CollectDimensionType(size_t dimNum, const std::string& equation, DimensionType2LabelInfo& labelInfo) const;
    void ReorderAxes(DimensionType2LabelInfo& labelInfos) const;
    void CompareAxes(EinsumDimensionType dimType, const DimensionType2LabelInfo& targetLabelInfo,
                     DimensionType2LabelInfo& inputLabelInfo) const;
    bool GetTransposeDstEquation(const std::string& oriEquation, const DimensionType2LabelInfo& labelInfos,
                                 std::vector<int32_t>& permList, bool& inputFreeContractOrder,
                                 std::string& dstEquation) const;
    void CheckMergeFreeLabels(const std::string& outEquation);
    void CheckBatchMatmulSwapInputs();

    // Static decomposition steps
    ge::Status TransposeInput(const std::vector<std::string>& inEquations, const ge::GNode& matchedNode);
    ge::Status StrideInput(const std::vector<std::string>& inEquations) const;
    ge::Status ReduceInput(const std::vector<std::string>& inEquations, const ge::GNode& matchedNode);
    ge::Status ReshapeInput(const std::vector<std::string>& inEquations, const std::string& outEquation,
                            const ge::GNode& matchedNode);
    void AppendMergedInputDims(std::vector<int64_t>& newDims, const DimensionType2LabelInfo& inputLabelInfo,
                               const std::vector<EinsumDimensionType>& checkOrders,
                               const std::vector<int64_t>& oriDims) const;
    ge::Status DoBatchMatmul(const std::vector<std::string>& inEquations, const ge::GNode& matchedNode);
    ge::Status ReshapeOutput(const ge::GNode& matchedNode, std::string& curOutEquation,
                             const std::vector<std::string>& inEquations);
    ge::Status InflatedOutput(const std::string& outEquation) const;
    ge::Status TransposeOutput(const std::string& outEquation, const ge::GNode& matchedNode,
                               std::string& curOutEquation);
    void CalcBatchMatmulOutput(size_t inputNum, const ge::GNode& matchedNode, bool& needReshape,
                               std::string& bmmOutEquation, std::vector<int64_t>& bmmDims) const;
    void CollectFreeLabelsAndDims(size_t inputNum, const ge::GNode& matchedNode, bool& needReshape,
                                  std::string& outFreeLabels, std::vector<int64_t>& outFreeDim) const;

    // Dynamic decomposition steps
    ge::Status InputLabelProcess(const std::string& lhs, size_t numOps, bool& ellInInput,
                                 std::vector<std::vector<uint8_t>>& opLabels);
    ge::Status LabelCountMap(const ge::GNode& matchedNode, size_t numOps, int64_t& ellNumDim,
                             std::vector<int64_t>& labelCount, std::vector<std::vector<uint8_t>>& opLabels,
                             size_t arrowPos);
    ge::Status ParseOutputLabel(const std::string& equation, size_t arrowPos, int64_t& ellNumDim, int64_t& permIndex,
                                bool& ellInOutput, std::vector<int64_t>& labelPermIndex,
                                std::vector<int64_t>& labelCount, int64_t& outNumDim, int64_t& ellIndex);
    ge::Status AlignInputDimForOutLabel(const ge::GNode& matchedNode, size_t numOps, int64_t& permIndex,
                                        std::vector<std::vector<uint8_t>>& opLabels, std::vector<int64_t>& dimCounts,
                                        std::vector<int64_t>& ellSizes, std::vector<int64_t>& labelSize,
                                        std::vector<int64_t>& labelPermIndex, int64_t& ellNumDim, int64_t& ellIndex);
    ge::Status ReduceSumInput(const ge::GNode& matchedNode, const std::vector<int64_t>& dimsToSum, size_t idx,
                              bool keepDims);

    // SumproductPair helpers (split from the original oversized function)
    struct DimClassification {
        std::vector<int> lro;
        std::vector<int> lo;
        std::vector<int> ro;
        int64_t lroSize = 1;
        int64_t loSize = 1;
        int64_t roSize = 1;
        int64_t sumSize = 1;
        std::vector<int64_t> leftSumDims;
        std::vector<int64_t> rightSumDims;
    };
    DimClassification ClassifySumproductDims(const std::shared_ptr<ge::TensorDesc>& leftDesc,
                                             const std::shared_ptr<ge::TensorDesc>& rightDesc,
                                             const std::vector<int64_t>& sumDims);
    void BuildSumproductPermutations(const DimClassification& cls, const std::vector<int64_t>& sumDims,
                                     int64_t outNumDim, std::vector<int>& lpermutation, std::vector<int>& rpermutation,
                                     std::vector<int>& opermutation) const;
    ge::Status BuildSumproductBmm(const ge::GNode& matchedNode, const DimClassification& cls,
                                  const std::vector<int64_t>& sumDims, int64_t outNumDim,
                                  std::shared_ptr<ge::TensorDesc>& rightDesc);

    ge::Status SumproductPair(const ge::GNode& matchedNode, const std::vector<int64_t>& sumDims, bool keepdim);
    ge::Status SumDimEmptyMul(const ge::GNode& matchedNode, std::shared_ptr<ge::TensorDesc>& leftDesc,
                              std::shared_ptr<ge::TensorDesc>& rightDesc, const std::vector<int64_t>& sumDims);
    ge::Status InsertGatherShape(std::shared_ptr<ge::OpDesc>& gatherShapesDesc,
                                 const std::vector<std::vector<int64_t>>& axes, std::vector<int64_t>& tmpDims,
                                 std::shared_ptr<ge::TensorDesc>& rightDesc);
    ge::Status CastNode(const ge::GNode& matchedNode, const ge::DataType& dtypeOut, int32_t idx);
    ge::Status PermuteInput(const ge::GNode& matchedNode, const std::vector<int>& permutation, size_t idx);
    ge::Status UnsqueezeInput(const ge::GNode& matchedNode, std::vector<int64_t> dims,
                              const std::vector<int64_t>& unsqueezeDim, int32_t idx);
    ge::Status PermuteReshapeInput(const ge::GNode& matchedNode, const std::vector<int>& permuteDim,
                                   const std::vector<int64_t>& reshapeDim, const std::vector<size_t>& size, size_t idx);
    ge::Status FinalizePermuteReshape(const ge::GNode& matchedNode, const std::vector<int64_t>& reshapeDim, size_t idx);
    ge::Status BatchMatmul(const ge::GNode& matchedNode);
    ge::Status ReshapePermuteInput(const ge::GNode& matchedNode, std::shared_ptr<ge::OpDesc>& gatherShapesDesc,
                                   const std::vector<int>& opermuteDim, const std::vector<int64_t>& outDim,
                                   const std::vector<int64_t>& unsqueezeDim);
    std::vector<int64_t> MaybeWrapDim(const std::vector<int64_t>& sumDims, int64_t dim) const;

    // Helpers to get prev output desc/node in the replacement chain
    std::shared_ptr<ge::TensorDesc> GetPrevOutputDesc(const ge::GNode& matchedNode, size_t idx) const;
    std::shared_ptr<ge::TensorDesc> GetPrevOutputDescAfterBmm(const ge::GNode& matchedNode) const;

    // Member state (reset per Replacement call)
    bool supportL12btBf16_ = false;
    bool swapBmmInputs_ = false;
    bool mergeFreeLabels_ = true;
    uint32_t castSeq_ = 1;
    uint32_t unsqueezeSeq_ = 1;
    uint32_t transposeSeq_ = 1;
    uint32_t reduceSeq_ = 1;
    uint32_t reshapeSeq_ = 1;
    uint32_t batchmatmulSeq_ = 1;
    std::map<char, EinsumDimensionType> dimTypesMap_;
    std::vector<bool> inputFreeContractOrders_;
    DimensionType2LabelInfo outputLabelInfo_;
    std::vector<DimensionType2LabelInfo> inputLabelInfos_;

    // Replacement graph builder state (for generic decomposition)
    ge::es::EsGraphBuilder* builder_ = nullptr;
    ge::Graph* graphPtr_ = nullptr;
    std::vector<ge::es::EsTensorHolder> subgraphInputHolders_;
    std::vector<std::vector<ge::GNode>> bmmInputNodes_;
    ge::GNode batchmatmulNode_;
    std::vector<ge::GNode> bmmOutputNodes_;
    ge::GNode gathershapeNode_;
    // Int32 inputs are forced to dynamic shape in dynamic decomposition
    // (static batchmatmul with int32 input fails to compile, same as legacy behavior)
    std::vector<std::vector<int64_t>> forcedInputDims_;

    // Dispatch table for the 28 specific dynamic equations (same as legacy dynamicShapeProcs_)
    using ProcFunc = ge::fusion::GraphUniqPtr (EinsumPass::*)(const std::vector<ge::fusion::SubgraphInput>&,
                                                              const ge::GNode&);
    static std::unordered_map<std::string, ProcFunc> dynamicShapeProcs_;

    void ResetState();
};

} // namespace ops

#endif // NN_EINSUM_FUSION_PASS_H
