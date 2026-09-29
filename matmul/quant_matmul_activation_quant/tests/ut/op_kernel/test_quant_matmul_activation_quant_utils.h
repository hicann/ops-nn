/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../../../tests/ut/common/ut_string_utils.h"

#ifdef __CCE_KT_TEST__
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_ut_runner.h"
#endif

namespace {
constexpr size_t GUARD_SIZE = 32;
constexpr uint8_t GUARD_VALUE = 0xA5;

struct QuantMatmulActivationQuantKernelTestParam {
    std::string socVersion;
    std::string caseName;
    std::string kernelUtTarget;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t activation = 0;
    uint32_t scaleAlg = 0;
    uint32_t outputDtype = 0;
    uint32_t inputDtype = 0;
    uint32_t weightDtype = 0;
    bool transB = false;
    bool fullLoad = false;
    uint32_t biasMode = 0;
    uint32_t blocks = 0;
    std::array<uint32_t, 4> batchA{};
    std::array<uint32_t, 4> batchB{};
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kL1 = 0;
    uint32_t scaleKL1 = 0;
    uint32_t nBufferNum = 0;
    uint32_t dbL0C = 0;
    uint32_t mTailTile = 1;
    uint32_t nTailTile = 1;
    std::string dataPattern;
    std::vector<uint8_t> expectedY;
    std::vector<uint8_t> expectedScale;
};

struct QuantMatmulActivationQuantKernelCsvLoadResult {
    std::vector<QuantMatmulActivationQuantKernelTestParam> params;
    std::vector<std::string> errors;
};

class QuantMatmulActivationQuantKernelTestUtils {
public:
    static QuantMatmulActivationQuantKernelCsvLoadResult GetParams(const std::string& socVersion,
                                                                   const std::string& testSuite,
                                                                   const std::string& csvPath = ResolveCsvPath())
    {
        QuantMatmulActivationQuantKernelCsvLoadResult result;
        std::ifstream csvData(csvPath);
        if (!csvData.is_open()) {
            result.errors.emplace_back("Cannot open kernel case file: " + csvPath);
            return result;
        }

        std::string line;
        size_t lineNo = 0;
        bool hasHeader = false;
        std::set<std::string> names;
        while (std::getline(csvData, line)) {
            ++lineNo;
            const auto trimmed = ut_str::Trim(line);
            if (trimmed.empty() || trimmed[0] == '#') {
                continue;
            }
            if (!hasHeader) {
                if (trimmed != CSV_HEADER) {
                    result.errors.emplace_back(csvPath + ":" + std::to_string(lineNo) + ": invalid CSV header");
                    return result;
                }
                hasHeader = true;
                continue;
            }

            std::vector<std::string> fields;
            ut_str::SplitStr2Vec(line, ",", fields);
            const std::string caseName = fields.size() > 1 ? ut_str::Trim(fields[1]) : "";
            try {
                auto param = ParseParam(fields);
                ValidateParam(param);
                if (!names.insert(param.caseName).second) {
                    throw std::runtime_error("duplicate caseName");
                }
                if (param.socVersion == socVersion && param.kernelUtTarget == testSuite) {
                    result.params.emplace_back(std::move(param));
                }
            } catch (const std::exception& error) {
                result.errors.emplace_back(csvPath + ":" + std::to_string(lineNo) + " [" + caseName +
                                           "]: " + error.what());
            }
        }
        if (result.params.empty()) {
            result.errors.emplace_back("No matching kernel cases for " + socVersion + "/" + testSuite + ": " + csvPath);
        }
        return result;
    }

private:
    static constexpr const char* CSV_HEADER =
        "socVersion,caseName,kernelUtTarget,m,n,k,activationType,scaleAlg,x1Dtype,x2Dtype,yDtype,transposeX2,"
        "fullLoad,biasMode,numBlocks,batchA,batchB,baseM,baseN,baseK,kL1,scaleKL1,nBufferNum,dbL0C,"
        "mTailTile,nTailTile,dataPattern,expectedY,expectedYScale,comment";

    static std::string ResolveCsvPath()
    {
        const std::string fileName = "test_quant_matmul_activation_quant.csv";
        const std::string relativePath = "matmul/quant_matmul_activation_quant/tests/ut/op_kernel/" + fileName;
        const std::string sourceFile = __FILE__;
        const auto pos = sourceFile.find_last_of("/\\");
        const std::vector<std::string> candidates = {sourceFile.substr(0, pos + 1) + fileName,
                                                     ut_str::GetExeDirPath() + "../../../../" + relativePath,
                                                     relativePath, fileName};
        for (const auto& path : candidates) {
            if (std::ifstream(path).good()) {
                return path;
            }
        }
        return candidates.front();
    }

    static uint32_t ParseU32(const std::string& value)
    {
        const auto token = ut_str::Trim(value);
        if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("expected an unsigned integer: " + token);
        }
        const uint64_t parsed = std::stoull(token);
        if (parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range("integer exceeds uint32_t: " + token);
        }
        return static_cast<uint32_t>(parsed);
    }

    static bool ParseBool(const std::string& value)
    {
        auto token = ut_str::Trim(value);
        std::transform(token.begin(), token.end(), token.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (token != "true" && token != "false" && token != "1" && token != "0") {
            throw std::runtime_error("expected true/false or 1/0: " + token);
        }
        return token == "true" || token == "1";
    }

    static std::array<uint32_t, 4> ParseBatch(const std::string& value)
    {
        std::istringstream input(value);
        std::array<uint32_t, 4> result{};
        std::string token;
        for (auto& dim : result) {
            if (!(input >> token) || (dim = ParseU32(token)) == 0) {
                throw std::runtime_error("batch must contain four positive dimensions");
            }
        }
        if (input >> token) {
            throw std::runtime_error("batch must contain exactly four dimensions");
        }
        return result;
    }

    static std::vector<uint8_t> ParseBytes(const std::string& value)
    {
        std::istringstream input(value);
        std::vector<uint8_t> result;
        std::string token;
        while (input >> token) {
            const auto byte = ParseU32(token);
            if (byte > std::numeric_limits<uint8_t>::max()) {
                throw std::out_of_range("expected output byte exceeds uint8_t: " + token);
            }
            result.emplace_back(static_cast<uint8_t>(byte));
        }
        return result;
    }

    static QuantMatmulActivationQuantKernelTestParam ParseParam(const std::vector<std::string>& fields)
    {
        if (fields.size() != 30) {
            throw std::runtime_error("invalid CSV column count; expected 30");
        }
        QuantMatmulActivationQuantKernelTestParam param;
        size_t index = 0;
        param.socVersion = ut_str::Trim(fields[index++]);
        param.caseName = ut_str::Trim(fields[index++]);
        param.kernelUtTarget = ut_str::Trim(fields[index++]);
        param.m = ParseU32(fields[index++]);
        param.n = ParseU32(fields[index++]);
        param.k = ParseU32(fields[index++]);
        const auto activation = ut_str::Trim(fields[index++]);
        const std::vector<std::string> activations = {"gelu_tanh", "gelu_erf", "swiglu"};
        const auto found = std::find(activations.begin(), activations.end(), activation);
        if (found == activations.end()) {
            throw std::runtime_error("invalid activationType: " + activation);
        }
        param.activation = static_cast<uint32_t>(found - activations.begin());
        param.scaleAlg = ParseU32(fields[index++]);
        param.inputDtype = ParseU32(fields[index++]);
        param.weightDtype = ParseU32(fields[index++]);
        param.outputDtype = ParseU32(fields[index++]);
        param.transB = ParseBool(fields[index++]);
        param.fullLoad = ParseBool(fields[index++]);
        param.biasMode = ParseU32(fields[index++]);
        param.blocks = ParseU32(fields[index++]);
        param.batchA = ParseBatch(fields[index++]);
        param.batchB = ParseBatch(fields[index++]);
        param.baseM = ParseU32(fields[index++]);
        param.baseN = ParseU32(fields[index++]);
        param.baseK = ParseU32(fields[index++]);
        param.kL1 = ParseU32(fields[index++]);
        param.scaleKL1 = ParseU32(fields[index++]);
        param.nBufferNum = ParseU32(fields[index++]);
        param.dbL0C = ParseU32(fields[index++]);
        param.mTailTile = ParseU32(fields[index++]);
        param.nTailTile = ParseU32(fields[index++]);
        param.dataPattern = ut_str::Trim(fields[index++]);
        param.expectedY = ParseBytes(fields[index++]);
        param.expectedScale = ParseBytes(fields[index++]);
        return param;
    }

    static uint32_t BatchCount(const std::array<uint32_t, 4>& shape)
    {
        uint64_t count = 1;
        for (const auto dim : shape) {
            count *= dim;
            if (count > std::numeric_limits<uint32_t>::max()) {
                throw std::out_of_range("batch count exceeds uint32_t");
            }
        }
        return static_cast<uint32_t>(count);
    }

    static bool IsWeightNz(const QuantMatmulActivationQuantKernelTestParam& param)
    {
        return param.kernelUtTarget == "QMMAQ_E4M3_WEIGHT_NZ";
    }

    static uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    static uint64_t WeightNzBatchStride(const QuantMatmulActivationQuantKernelTestParam& param)
    {
        constexpr uint64_t c0 = 32;
        constexpr uint64_t blockCube = 16;
        if (param.transB) {
            return AlignUp(param.k, c0) * param.n;
        }
        return param.n * AlignUp(param.k, blockCube);
    }

    static uint64_t WeightNzOffset(const QuantMatmulActivationQuantKernelTestParam& param, uint32_t k, uint32_t n)
    {
        constexpr uint64_t c0 = 32;
        constexpr uint64_t blockCube = 16;
        if (param.transB) {
            // ZN backing order: [ceil(K/32), ceil(N/16), 16, 32].
            return (k % c0) + (k / c0) * c0 * param.n + (n % blockCube) * c0 + (n / blockCube) * blockCube * c0;
        }
        // NZ backing order: [ceil(N/32), ceil(K/16), 16, 32].
        const uint64_t alignedK = AlignUp(param.k, blockCube);
        return (k % blockCube) * c0 + (k / blockCube) * blockCube * c0 + (n % c0) + (n / c0) * c0 * alignedK;
    }

    static void ValidateParam(const QuantMatmulActivationQuantKernelTestParam& param)
    {
        if (param.caseName.empty() ||
            param.caseName.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
                std::string::npos) {
            throw std::runtime_error("caseName must be a nonempty GoogleTest parameter name");
        }
        const bool e4m3 = param.kernelUtTarget == "QMMAQ_E4M3" && param.inputDtype == 36 && param.outputDtype == 36;
        const bool e5m2 = param.kernelUtTarget == "QMMAQ_E5M2" && param.inputDtype == 36 && param.outputDtype == 35;
        const bool e5m2Gelu = param.kernelUtTarget == "QMMAQ_E5M2_GELU" && param.inputDtype == 35 &&
                              param.outputDtype == 35;
        const bool e4m3WeightNz = IsWeightNz(param) && param.inputDtype == 36 && param.weightDtype == 36 &&
                                  param.outputDtype == 36;
        const bool ndDtypes = (e4m3 || e5m2 || e5m2Gelu) && param.weightDtype == 35;
        if (param.socVersion != "Ascend950" || !(ndDtypes || e4m3WeightNz)) {
            throw std::runtime_error("socVersion/kernelUtTarget/dtypes do not match a compiled kernel specialization");
        }
        if (IsWeightNz(param) && param.activation != 2) {
            throw std::runtime_error("WeightNZ kernel cases must use SwiGLU");
        }
        if (param.m == 0 || param.n == 0 || param.k == 0 || param.blocks == 0 || param.scaleAlg > 1 ||
            param.biasMode > 2 || (param.activation == 2 && param.n % 64 != 0) ||
            (param.activation != 2 && param.inputDtype != param.outputDtype)) {
            throw std::runtime_error("invalid shape, activation, scaleAlg, biasMode or numBlocks");
        }
        std::array<uint32_t, 4> batchC{};
        for (size_t dim = 0; dim < batchC.size(); ++dim) {
            if (param.batchA[dim] != param.batchB[dim] && param.batchA[dim] != 1 && param.batchB[dim] != 1) {
                throw std::runtime_error("batch shapes cannot broadcast");
            }
            batchC[dim] = std::max(param.batchA[dim], param.batchB[dim]);
        }
        const uint64_t batches = BatchCount(batchC);
        const uint64_t width = param.activation == 2 ? param.n / 2 : param.n;
        if (param.dataPattern != "batch_constant" && param.dataPattern != "coordinate" &&
            param.dataPattern != "subnormal_bias" && param.dataPattern != "gate_subnormal") {
            throw std::runtime_error("invalid dataPattern");
        }
        if ((param.dataPattern == "subnormal_bias" || param.dataPattern == "gate_subnormal") &&
            (param.activation != 2 || param.biasMode == 0)) {
            throw std::runtime_error("subnormal_bias/gate_subnormal requires SwiGLU with bias");
        }
        const uint64_t expectedRows = batches * (param.dataPattern == "coordinate" ? param.m : 1);
        if (param.expectedY.size() != expectedRows * width ||
            param.expectedScale.size() != expectedRows * ((width + 63) / 64 * 2)) {
            throw std::runtime_error("expectedY/expectedYScale size does not match dataPattern");
        }
        if (param.fullLoad && batches != 1) {
            throw std::runtime_error("full-load test specialization requires one output batch");
        }
        // CSV controls tile sizes and QBMM's 2/3/4 L1 data buffers; scales/bias remain double-buffered.
        if (std::max({param.baseM, param.baseN, param.baseK}) > std::numeric_limits<uint16_t>::max() ||
            param.baseM == 0 || param.baseM % 16 != 0 || param.baseN == 0 || param.baseN % 128 != 0 ||
            param.baseK == 0 || param.baseK % 64 != 0 || param.kL1 == 0 || param.kL1 % param.baseK != 0 ||
            param.scaleKL1 == 0 || param.scaleKL1 % param.kL1 != 0 || param.nBufferNum < 2 || param.nBufferNum > 4 ||
            (param.dbL0C != 1 && param.dbL0C != 2) || param.mTailTile == 0 || param.nTailTile == 0 ||
            std::max(param.mTailTile, param.nTailTile) > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("invalid baseM/baseN/baseK/kL1/scaleKL1/nBufferNum/dbL0C/mTailTile/nTailTile");
        }
        uint64_t resourceBaseN = param.baseN;
        if (IsWeightNz(param)) {
            const uint64_t halfN = param.n / 2;
            const uint64_t outputBaseN = std::min<uint64_t>(param.baseN / 2, halfN);
            resourceBaseN = outputBaseN * 2;
        }
        const uint64_t tileArea = static_cast<uint64_t>(param.baseM) * resourceBaseN;
        const uint64_t biasBytes = param.biasMode == 0 ? 0 : resourceBaseN * sizeof(float);
        uint64_t dataBytes = resourceBaseN * param.kL1;
        uint64_t scaleBiasBytes = resourceBaseN * (param.scaleKL1 / 32) + biasBytes;
        uint64_t fullLoadBytes = 0;
        if (param.fullLoad) {
            const uint64_t alignedK = (static_cast<uint64_t>(param.k) + 63) / 64 * 64;
            fullLoadBytes = static_cast<uint64_t>(param.baseM) * (alignedK + alignedK / 32);
        } else {
            dataBytes += static_cast<uint64_t>(param.baseM) * param.kL1;
            scaleBiasBytes += static_cast<uint64_t>(param.baseM) * (param.scaleKL1 / 32);
        }
        const uint64_t frontBytes = dataBytes * ((param.nBufferNum + 1) / 2) + scaleBiasBytes + fullLoadBytes;
        const uint64_t backBytes = dataBytes * (param.nBufferNum / 2) + scaleBiasBytes;
        const uint64_t l1Bytes = std::max<uint64_t>(256 * 1024, frontBytes) + backBytes;
        const uint64_t inputUbElements = AlignUp(param.baseM, 2) * resourceBaseN;
        if (tileArea * sizeof(float) > 128 * 1024 || l1Bytes > 512 * 1024 || biasBytes * 2 > 4096 ||
            std::max<uint64_t>(param.baseM, resourceBaseN) * param.baseK > 32 * 1024 ||
            (IsWeightNz(param) && inputUbElements > 2 * 64 * 256)) {
            throw std::runtime_error("tiling exceeds the double-buffered Cube capacity");
        }
    }

#ifdef __CCE_KT_TEST__
    class GmBuffer {
    public:
        explicit GmBuffer(size_t bytes) : bytes_(bytes)
        {
            allocation_ = static_cast<uint8_t*>(AscendC::GmAlloc(bytes + 2 * GUARD_SIZE));
            if (allocation_ == nullptr) {
                throw std::bad_alloc();
            }
            std::memset(allocation_, GUARD_VALUE, bytes + 2 * GUARD_SIZE);
        }

        ~GmBuffer() { AscendC::GmFree(allocation_); }

        GmBuffer(const GmBuffer&) = delete;
        GmBuffer& operator=(const GmBuffer&) = delete;

        uint8_t* Data() const { return allocation_ + GUARD_SIZE; }

        void CheckGuards() const
        {
            for (size_t index = 0; index < GUARD_SIZE; ++index) {
                EXPECT_EQ(allocation_[index], GUARD_VALUE) << "prefix guard " << index;
                EXPECT_EQ(Data()[bytes_ + index], GUARD_VALUE) << "suffix guard " << index;
            }
        }

    private:
        size_t bytes_;
        uint8_t* allocation_;
    };

    static QMMAQ::QMMAQTilingData MakeTiling(const QuantMatmulActivationQuantKernelTestParam& item)
    {
        QMMAQ::QMMAQTilingData data{};
        std::array<uint32_t, 4> batchC{};
        for (size_t index = 0; index < batchC.size(); ++index) {
            batchC[index] = std::max(item.batchA[index], item.batchB[index]);
        }
        data.m = item.m;
        data.n = item.n;
        data.k = item.k;
        data.kL1 = item.kL1;
        data.scaleKL1 = item.scaleKL1;
        data.batchA1 = item.batchA[0];
        data.batchA2 = item.batchA[1];
        data.batchA3 = item.batchA[2];
        data.batchA4 = item.batchA[3];
        data.batchB1 = item.batchB[0];
        data.batchB2 = item.batchB[1];
        data.batchB3 = item.batchB[2];
        data.batchB4 = item.batchB[3];
        data.batchC1 = batchC[0];
        data.batchC2 = batchC[1];
        data.batchC3 = batchC[2];
        data.batchC4 = batchC[3];
        data.batchCount = BatchCount(batchC);
        data.baseM = static_cast<uint16_t>(item.baseM);
        data.baseN = static_cast<uint16_t>(item.baseN);
        data.baseK = static_cast<uint16_t>(item.baseK);
        data.mTailTile = static_cast<uint16_t>(item.mTailTile);
        data.nTailTile = static_cast<uint16_t>(item.nTailTile);
        data.nBufferNum = static_cast<uint8_t>(item.nBufferNum);
        data.isBias = item.biasMode != 0;
        data.dbL0C = static_cast<uint8_t>(item.dbL0C);
        data.biasThreeDim = item.biasMode == 2;
        data.activationType = static_cast<QMMAQ::ActivationAlg>(item.activation);
        data.scaleAlg = static_cast<QMMAQ::QuantAlg>(item.scaleAlg);
        return data;
    }

    using Entry = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

    template <int BATCHMODE>
    static Entry GetKernelEntry(const QuantMatmulActivationQuantKernelTestParam& item)
    {
        // [SwiGLU][transposeX2][A full-load]. Match host kernel-type dispatch for each CSV case.
        constexpr Entry entries[2][2][2] = {{{quant_matmul_activation_quant<0, 0, BATCHMODE, TPL_GELU_NO_FULLLOAD>,
                                              quant_matmul_activation_quant<0, 0, BATCHMODE, TPL_GELU_FULLLOAD>},
                                             {quant_matmul_activation_quant<0, 1, BATCHMODE, TPL_GELU_NO_FULLLOAD>,
                                              quant_matmul_activation_quant<0, 1, BATCHMODE, TPL_GELU_FULLLOAD>}},
                                            {{quant_matmul_activation_quant<0, 0, BATCHMODE, TPL_SWIGLU_NO_FULLLOAD>,
                                              quant_matmul_activation_quant<0, 0, BATCHMODE, TPL_SWIGLU_FULLLOAD>},
                                             {quant_matmul_activation_quant<0, 1, BATCHMODE, TPL_SWIGLU_NO_FULLLOAD>,
                                              quant_matmul_activation_quant<0, 1, BATCHMODE, TPL_SWIGLU_FULLLOAD>}}};
        return entries[item.activation == 2][item.transB][item.fullLoad];
    }

public:
    static void TestOneParamCase950(const QuantMatmulActivationQuantKernelTestParam& item)
    {
        ValidateParam(item);
        const auto data = MakeTiling(item);
        const uint32_t batchA = BatchCount(item.batchA);
        const uint32_t batchB = BatchCount(item.batchB);
        const uint32_t batchC = data.batchCount;
        const bool withoutBatch = batchA == 1 && batchB == 1;
        const QMMAQ::QMMAQWithoutBatchTilingData withoutBatchData{data.m,
                                                                  data.n,
                                                                  data.k,
                                                                  data.kL1,
                                                                  data.scaleKL1,
                                                                  data.dstTypeMax,
                                                                  data.baseM,
                                                                  data.baseN,
                                                                  data.baseK,
                                                                  data.mTailTile,
                                                                  data.nTailTile,
                                                                  data.mBaseTailSplitCnt,
                                                                  data.nBaseTailSplitCnt,
                                                                  data.mTailMain,
                                                                  data.nTailMain,
                                                                  data.nBufferNum,
                                                                  data.isBias,
                                                                  data.dbL0C,
                                                                  data.weightMustHitL2,
                                                                  data.activationType,
                                                                  data.scaleAlg,
                                                                  data.roundMode};
        const uint32_t width = item.activation == 2 ? item.n / 2 : item.n;
        const uint32_t scaleStride = (width + 63) / 64 * 2;
        const size_t outputBytes = static_cast<size_t>(batchC) * item.m * width;
        const uint32_t kGroups = (item.k + 63) / 64;
        const bool weightNz = IsWeightNz(item);
        const uint64_t x2BatchStride = weightNz ? WeightNzBatchStride(item) : static_cast<uint64_t>(item.k) * item.n;
        GmBuffer x1(static_cast<size_t>(batchA) * item.m * item.k);
        GmBuffer x2(static_cast<size_t>(batchB) * x2BatchStride);
        GmBuffer x1Scale(static_cast<size_t>(batchA) * item.m * kGroups * 2);
        GmBuffer x2Scale(static_cast<size_t>(batchB) * item.n * kGroups * 2);
        const uint32_t biasBatches = item.biasMode == 2 ? batchC : 1;
        GmBuffer bias(static_cast<size_t>(biasBatches) * item.n * sizeof(float));
        GmBuffer y(outputBytes);
        GmBuffer yScale(static_cast<size_t>(batchC) * item.m * scaleStride);
        GmBuffer workspace(16 * 1024 * 1024);
        const size_t tilingBytes = withoutBatch ? sizeof(withoutBatchData) : sizeof(data);
        const void* tilingData = withoutBatch ? static_cast<const void*>(&withoutBatchData) :
                                                static_cast<const void*>(&data);
        GmBuffer tiling(tilingBytes);
        std::memcpy(tiling.Data(), tilingData, tilingBytes);
        std::memset(x2.Data(), 0, static_cast<size_t>(batchB) * x2BatchStride);

        const std::array<uint8_t, 3> aValues = item.inputDtype == 35 ? std::array<uint8_t, 3>{0x30, 0x34, 0x36} :
                                                                       std::array<uint8_t, 3>{0x20, 0x28, 0x2C};
        const std::array<uint8_t, 3> bLeft = item.weightDtype == 36 ? std::array<uint8_t, 3>{0x20, 0x28, 0x2C} :
                                                                      std::array<uint8_t, 3>{0x30, 0x34, 0x36};
        const std::array<uint8_t, 3> bRight = item.weightDtype == 36 ? std::array<uint8_t, 3>{0x28, 0x30, 0x34} :
                                                                       std::array<uint8_t, 3>{0x34, 0x38, 0x3A};
        const std::array<uint8_t, 3> variedA = item.inputDtype == 35 ? std::array<uint8_t, 3>{0x28, 0x2C, 0x2E} :
                                                                       std::array<uint8_t, 3>{0x10, 0x18, 0x1C};
        const bool varied = item.dataPattern == "coordinate";
        const bool subnormal = item.dataPattern == "subnormal_bias";
        const bool gateSubnormal = item.dataPattern == "gate_subnormal";
        for (uint32_t batch = 0; batch < batchA; ++batch) {
            for (uint32_t row = 0; row < item.m; ++row) {
                const size_t rowIndex = static_cast<size_t>(batch) * item.m + row;
                for (uint32_t k = 0; k < item.k; ++k) {
                    x1.Data()[rowIndex * item.k + k] = (subnormal || gateSubnormal) ?
                                                           0 :
                                                           (varied ? variedA[(batch + row + 2 * k) % 3] :
                                                                     aValues[batch % 3]);
                }
                for (uint32_t group = 0; group < kGroups * 2; ++group) {
                    x1Scale.Data()[rowIndex * kGroups * 2 + group] = varied ? 126 + (batch + row + group) % 3 :
                                                                              127 + batch % 2;
                }
            }
        }
        for (uint32_t batch = 0; batch < batchB; ++batch) {
            for (uint32_t k = 0; k < item.k; ++k) {
                for (uint32_t n = 0; n < item.n; ++n) {
                    const size_t offset = static_cast<size_t>(batch) * x2BatchStride +
                                          (weightNz ? WeightNzOffset(item, k, n) :
                                                      (item.transB ? static_cast<size_t>(n) * item.k + k :
                                                                     static_cast<size_t>(k) * item.n + n));
                    const uint32_t valueIndex = varied ? (batch + 2 * n + k) % 3 : batch % 3;
                    x2.Data()[offset] = n < item.n / 2 ? bLeft[valueIndex] : bRight[valueIndex];
                }
            }
            for (uint32_t n = 0; n < item.n; ++n) {
                for (uint32_t group = 0; group < kGroups * 2; ++group) {
                    const size_t offset = static_cast<size_t>(batch) * item.n * kGroups * 2 +
                                          (item.transB ?
                                               static_cast<size_t>(n) * kGroups * 2 + group :
                                               static_cast<size_t>(group / 2) * item.n * 2 + n * 2 + group % 2);
                    x2Scale.Data()[offset] = varied ? 126 + (batch + n + 2 * group) % 3 : 127 - batch % 2;
                }
            }
        }
        auto* biasData = reinterpret_cast<float*>(bias.Data());
        for (uint32_t batch = 0; batch < biasBatches; ++batch) {
            for (uint32_t n = 0; n < item.n; ++n) {
                if (subnormal) {
                    // Normal FP32 bias values yield nonzero BF16 subnormals after SiLU(gate=1) * linear.
                    const uint32_t group = n < item.n / 2 ? 0 : (n - item.n / 2) / 32;
                    biasData[static_cast<size_t>(batch) * item.n + n] = n < item.n / 2 ?
                                                                            1.0F :
                                                                            (group % 3 == 2 ?
                                                                                 0.0F :
                                                                                 (group % 3 == 0 ? 1.0F : -1.0F) *
                                                                                     std::numeric_limits<float>::min());
                    continue;
                }
                if (gateSubnormal) {
                    // FP32 subnormal gate values are flushed to signed zero by the kernel's
                    // FTZ Div, so SiLU(gate=subnormal) * linear(1.0) yields an all-zero gluRes and
                    // zero scale groups. The golden models the same FTZ boundary.
                    biasData[static_cast<size_t>(batch) * item.n + n] = n < item.n / 2 ?
                                                                            std::numeric_limits<float>::denorm_min() :
                                                                            1.0F;
                    continue;
                }
                biasData[static_cast<size_t>(batch) * item.n + n] = n < item.n / 2 ?
                                                                        0.125F + batch * 0.25F + (n % 7) * 0.03125F :
                                                                        -0.5F + ((n - item.n / 2) % 5) * 0.0625F;
            }
        }
        const Entry entry = withoutBatch ? GetKernelEntry<TPL_WITHOUT_BATCH>(item) :
                                           GetKernelEntry<TPL_WITH_BATCH>(item);
        uint8_t* biasAddress = item.biasMode == 0 ? nullptr : bias.Data();
        ASSERT_TRUE(KERNEL_RUN_KF(entry, item.blocks, x1.Data(), x2.Data(), biasAddress, x1Scale.Data(), x2Scale.Data(),
                                  y.Data(), yScale.Data(), workspace.Data(), tiling.Data()))
            << "Kernel CPU simulator failed: " << item.caseName;
        for (uint32_t batch = 0; batch < batchC; ++batch) {
            for (uint32_t row = 0; row < item.m; ++row) {
                const size_t expectedRow = varied ? static_cast<size_t>(batch) * item.m + row : batch;
                for (uint32_t n = 0; n < width; ++n) {
                    const size_t expected = expectedRow * width + n;
                    const size_t offset = (static_cast<size_t>(batch) * item.m + row) * width + n;
                    EXPECT_EQ(y.Data()[offset], item.expectedY.at(expected)) << "y offset " << offset;
                }
                for (uint32_t group = 0; group < scaleStride; ++group) {
                    const size_t expected = expectedRow * scaleStride + group;
                    const uint8_t value = item.expectedScale.at(expected);
                    const size_t offset = (static_cast<size_t>(batch) * item.m + row) * scaleStride + group;
                    EXPECT_EQ(yScale.Data()[offset], value) << "scale offset " << offset;
                }
            }
        }
        tiling.CheckGuards();
        x2.CheckGuards();
        y.CheckGuards();
        yScale.CheckGuards();
    }

#endif
};

} // namespace
