/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_inplace_index_add_with_sorted.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_func_def.h"
#include "data_utils.h"

#define IS_CAST_FLOAT ((is_same<T, half>::value) || (is_same<T, bfloat16_t>::value))

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {
};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {
};

struct InplaceIndexAddWithSortedCompileInfo {
    int32_t totalCoreNum = 40;
    uint64_t ubSizePlatForm = 0;
    uint64_t workspaceSize = 0;
};

extern "C" __global__ __aicore__ void inplace_index_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos, GM_ADDR alpha, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling);

std::string GetShapesString(const std::vector<std::vector<int64_t>>& shapeInfo)
{
    std::string ret = "{";
    for (auto shape : shapeInfo) {
        ret += "{";
        for (auto dim : shape) {
            ret += std::to_string(dim) + ",";
        }
        ret += "},";
    }
    return ret + "}";
}

std::string GetShapesString(const std::vector<int64_t>& shape)
{
    std::string ret = "{";
    for (auto dim : shape) {
        ret += std::to_string(dim) + ",";
    }
    return ret + "}";
}

int64_t GetShapeSize(const std::vector<std::vector<int64_t>>& shapeInfo, const size_t& index)
{
    int64_t shapeSize = 1;
    for (auto shape : shapeInfo[index]) {
        shapeSize *= shape;
    }
    return shapeSize;
}

class inplace_index_add_with_sorted_test : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        std::cout << "inplace_index_add_with_sorted_test SetUpTestSuite" << std::endl;
    }

    static void TearDownTestSuite()
    {
        std::cout << "inplace_index_add_with_sorted_test SetUpTestSuite" << std::endl;
    }

    template <typename T>
    void SingleCallOperator(
        const std::vector<std::vector<int64_t>>& shapeInfos, const std::vector<int64_t>& attrInfos,
        const ge::DataType varDtype)
    {
        // get tiling data and tiling key
        optiling::InplaceIndexAddWithSortedTilingDef tilingObject(shapeInfos, attrInfos, varDtype);
        tilingObject.RunKernelTiling();
        int32_t tilingKey = tilingObject.GetTilingKey();

        system(
            "cp -r "
            "../../../../index/inplace_index_add_with_sorted/tests/ut/op_kernel/"
            "inplace_index_add_with_sorted_data ./");
        system(
            "chmod -R 755 ./inplace_index_add_with_sorted_data/ && rm -rf ./inplace_index_add_with_sorted_data/*bin");
        std::string genCMD = "cd ./inplace_index_add_with_sorted_data/ && python3 gen_data.py '" +
                             GetShapesString(shapeInfos) + "' '" + GetShapesString(attrInfos) + "' " +
                             std::to_string(tilingKey);
        system(genCMD.c_str());
        size_t usrWorkspaceSize = 4096;
        uint8_t* usrWorkSpace = (uint8_t*)AscendC::GmAlloc(usrWorkspaceSize);
        size_t tilingSize = sizeof(InplaceIndexAddWithSortedTilingData);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
        tilingObject.TilingDataSet(tiling);

        size_t varByteSize = GetShapeSize(shapeInfos, 0) * sizeof(T);
        size_t valueByteSize = GetShapeSize(shapeInfos, 1) * sizeof(T);
        size_t indexByteSize = GetShapeSize(shapeInfos, 2) * sizeof(int32_t);
        size_t alphaByteSize = sizeof(T);
        if constexpr (IS_CAST_FLOAT) {
            alphaByteSize = sizeof(float);
        }
        size_t outByteSize = varByteSize;

        uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
        uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueByteSize);
        uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexByteSize);
        uint8_t* pos = (uint8_t*)AscendC::GmAlloc(indexByteSize);
        uint8_t* alpha = (uint8_t*)AscendC::GmAlloc(alphaByteSize);
        uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

        ReadFile("./inplace_index_add_with_sorted_data/var.bin", varByteSize, var, varByteSize);
        ReadFile("./inplace_index_add_with_sorted_data/value.bin", valueByteSize, value, valueByteSize);
        ReadFile("./inplace_index_add_with_sorted_data/index.bin", indexByteSize, index, indexByteSize);
        ReadFile("./inplace_index_add_with_sorted_data/pos.bin", indexByteSize, pos, indexByteSize);
        ReadFile("./inplace_index_add_with_sorted_data/alpha.bin", alphaByteSize, alpha, alphaByteSize);

        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(
            inplace_index_add_with_sorted, tilingObject.GetNeedCoreNum(), var, value, index, pos, alpha, out,
            usrWorkSpace, tiling);

        WriteFile("./inplace_index_add_with_sorted_data/output_dx.bin", var, varByteSize);

        AscendC::GmFree((void*)var);
        AscendC::GmFree((void*)value);
        AscendC::GmFree((void*)index);
        AscendC::GmFree((void*)pos);
        AscendC::GmFree((void*)alpha);
        AscendC::GmFree((void*)out);
        AscendC::GmFree((void*)tiling);
        AscendC::GmFree((void*)usrWorkSpace);
    }
};

TEST_F(inplace_index_add_with_sorted_test, test_case_bfloat16_key_3)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2048, 200}, {1024, 200}, {1024}});
    std::vector<int64_t> attrInfos({0});
    SingleCallOperator<bfloat16_t>(shapeInfos, attrInfos, ge::DT_BF16);
}

TEST_F(inplace_index_add_with_sorted_test, test_case_float16_key_2)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2048, 200}, {1024, 200}, {1024}});
    std::vector<int64_t> attrInfos({0});
    SingleCallOperator<half>(shapeInfos, attrInfos, ge::DT_FLOAT16);
}

TEST_F(inplace_index_add_with_sorted_test, test_case_float_key_1)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2048, 200}, {204, 200}, {204}});
    std::vector<int64_t> attrInfos({0});
    SingleCallOperator<float>(shapeInfos, attrInfos, ge::DT_FLOAT);
}

TEST_F(inplace_index_add_with_sorted_test, test_case_int16_key_4)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2048, 200}, {102, 200}, {102}});
    std::vector<int64_t> attrInfos({0});
    SingleCallOperator<int16_t>(shapeInfos, attrInfos, ge::DT_INT16);
}

TEST_F(inplace_index_add_with_sorted_test, test_case_int32_key_5)
{
    std::vector<std::vector<int64_t>> shapeInfos({{4096, 1000}, {100, 1000}, {100}});
    std::vector<int64_t> attrInfos({0});
    SingleCallOperator<int32_t>(shapeInfos, attrInfos, ge::DT_INT32);
}