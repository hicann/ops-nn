/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <random>

#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return 0;
}

std::vector<uint16_t> GenerateRandomBf16Data(int64_t size, unsigned int seed = 42)
{
    std::vector<uint16_t> data(size);
    std::mt19937 gen(seed);
    for (int64_t i = 0; i < size; i++) {
        int sign = (gen() % 2) ? 0x8000 : 0;
        int exp = 0x3F00 + (gen() % 2);
        int mant = gen() % 128;
        data[i] = sign | exp | mant;
    }
    return data;
}

std::vector<uint16_t> GenerateIdentityMatrix(int64_t K)
{
    std::vector<uint16_t> matrix(K * K, 0);
    uint16_t bf16One = 0x3F80;
    for (int64_t i = 0; i < K; i++) {
        matrix[i * K + i] = bf16One;
    }
    return matrix;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    {
        // V1普通RotateQuant：根据芯片选择受支持的输出类型和scale布局。
        const char* socName = aclrtGetSocName();
        CHECK_RET(socName != nullptr, LOG_PRINT("aclrtGetSocName failed.\n"); return 1);
        std::string soc(socName);
        std::transform(soc.begin(), soc.end(), soc.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const bool isInt = soc.rfind("ascend910b", 0) == 0 || soc.rfind("ascend910_93", 0) == 0;
        const bool isMx = soc.rfind("ascend950", 0) == 0 || soc.rfind("ascend350", 0) == 0 ||
                          soc.rfind("ascend960dt", 0) == 0 || soc.rfind("ascend960pr", 0) == 0;
        CHECK_RET(isInt || isMx, LOG_PRINT("Unsupported SOC for this example: %s\n", socName); return 1);
        LOG_PRINT("Current SOC: %s; mode=ordinary, y=%s, scale=%s\n", socName, isMx ? "FLOAT8_E4M3FN" : "INT8",
                  isMx ? "FLOAT8_E8M0" : "FLOAT");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        // 所有输入行共用一个二维旋转矩阵，取K=N，直接计算x @ rotation后量化。
        // x: (M, N), rot: (K, K), y: (M, N)
        // INT scale: (M,); MX scale: (M, CeilDiv(N, 64), 2)
        const int64_t M = 1024;
        const int64_t N = 256;
        const int64_t K = 64;

        std::vector<int64_t> xShape = {M, N};
        std::vector<int64_t> rotShape = {K, K};
        std::vector<int64_t> yShape = {M, N};
        std::vector<int64_t> scaleShape = isMx ? std::vector<int64_t>{M, (N + 63) / 64, 2} : std::vector<int64_t>{M};

        // 创建x aclTensor (BF16)
        auto xHostData = GenerateRandomBf16Data(M * N, 42);
        void* xDeviceAddr = nullptr;
        aclTensor* xTensor = nullptr;
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &xTensor);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> xAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create x tensor failed.\n"); return ret);

        // 创建rot aclTensor (BF16, 必须为方阵)
        auto rotHostData = GenerateIdentityMatrix(K);
        void* rotDeviceAddr = nullptr;
        aclTensor* rotTensor = nullptr;
        ret = CreateAclTensor(rotHostData, rotShape, &rotDeviceAddr, aclDataType::ACL_BF16, &rotTensor);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rotTensorPtr(rotTensor, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> rotAddrPtr(rotDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create rot tensor failed.\n"); return ret);

        // 创建y aclTensor：INT8和FLOAT8_E4M3FN每个元素均占一个字节。
        std::vector<int8_t> yHostData(M * N, 0);
        void* yDeviceAddr = nullptr;
        aclTensor* yTensor = nullptr;
        ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr,
                              isMx ? aclDataType::ACL_FLOAT8_E4M3FN : aclDataType::ACL_INT8, &yTensor);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> yAddrPtr(yDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create y tensor failed.\n"); return ret);

        // 创建scale aclTensor：INT模式使用FLOAT，MX模式使用一字节的FLOAT8_E8M0原始编码。
        auto scaleSize = GetShapeSize(scaleShape);
        std::vector<float> scaleHostData(isMx ? 0 : scaleSize, 0.0f);
        std::vector<uint8_t> scaleMxHostData(isMx ? scaleSize : 0, 0);
        void* scaleDeviceAddr = nullptr;
        aclTensor* scaleTensor = nullptr;
        if (isMx) {
            ret = CreateAclTensor(scaleMxHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                  &scaleTensor);
        } else {
            ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleTensor);
        }
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scaleTensor, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> scaleAddrPtr(scaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create scale tensor failed.\n"); return ret);

        // 3. 调用CANN算子库API，需要修改为具体的Api名称
        // 调用V1第一段接口执行普通旋转量化，alpha传nullptr关闭clamp。
        int64_t axis = -1;
        char roundMode[] = "rint";
        int64_t scaleAlg = 0;
        double dstTypeMax = 0.0;
        bool trans = false;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnRotateQuantGetWorkspaceSize(xTensor, rotTensor, nullptr, axis, roundMode, scaleAlg, dstTypeMax,
                                               trans, yTensor, scaleTensor, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotateQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);

        // 根据第一段接口计算出的workspaceSize申请device内存
        void* workspaceAddr = nullptr;
        std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
            workspaceAddrPtr.reset(workspaceAddr);
        }

        // 调用aclnnRotateQuant第二段接口
        ret = aclnnRotateQuant(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotateQuant failed. ERROR: %d\n", ret); return ret);

        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
        auto ySize = GetShapeSize(yShape);
        std::vector<int8_t> yResult(ySize, 0);
        ret = aclrtMemcpy(yResult.data(), ySize * sizeof(int8_t), yDeviceAddr, ySize * sizeof(int8_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed.\n"); return ret);

        void* scaleResult = isMx ? static_cast<void*>(scaleMxHostData.data()) :
                                   static_cast<void*>(scaleHostData.data());
        auto scaleBytes = scaleSize * (isMx ? sizeof(uint8_t) : sizeof(float));
        ret = aclrtMemcpy(scaleResult, scaleBytes, scaleDeviceAddr, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result failed.\n"); return ret);

        // 打印部分结果
        constexpr int64_t kPrintLimit = 5;
        for (int64_t i = 0; i < kPrintLimit && i < static_cast<int64_t>(yResult.size()); i++) {
            if (isMx) {
                // FP8/E8M0仅打印原始编码，不能直接按INT8/FLOAT解释。
                LOG_PRINT("y[%ld] code = 0x%02x\n", i, static_cast<unsigned>(static_cast<uint8_t>(yResult[i])));
            } else {
                LOG_PRINT("y[%ld] = %d\n", i, static_cast<int>(yResult[i]));
            }
        }
        for (int64_t i = 0; i < kPrintLimit && i < scaleSize; i++) {
            if (isMx) {
                LOG_PRINT("scale[%ld] code = 0x%02x\n", i, static_cast<unsigned>(scaleMxHostData[i]));
            } else {
                LOG_PRINT("scale[%ld] = %f\n", i, scaleHostData[i]);
            }
        }
    }

    // 6. 释放资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
