/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_ascend_anti_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 将16位half位模式转换为32位float
float half_to_float(uint16_t h)
{
    // 分别提取half的符号、指数和尾数
    int sign = (h >> 15) & 0x001;
    int exp = (h >> 10) & 0x01F;
    int mant = h & 0x03FF;

    // 如果指数为0，表示0或次正规数
    // 如果指数为31，表示无穷大或NaN
    // 否则，指数需要减去15的偏置，并加上127的float偏置
    int f_exp;
    int f_mant;

    if (exp == 0) {
        // 0或次正规数
        if (mant == 0) {
            // 0
            f_exp = 0;
            f_mant = 0;
        } else {
            // 次正规数，需要将尾数规范化，指数固定为-14
            // 在float中，指数为-127，所以float的指数为-127+(-14) = -141
            // 但是，float的次正规数指数固定为-127，所以这里我们直接计算尾数
            // 注意：half的次正规数指数是-14，尾数不包含隐含的1
            // 将half的次正规数转换为float的次正规数
            f_exp = 0;
            // 将half的10位尾数左移13位，变成23位尾数，但是需要规范化
            // 由于是次正规数，我们需要将尾数左移直到第一个1出现，然后调整指数
            int shift = 0;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x03FF;
            f_exp = -14 - shift + 127; // 加上float的偏置127
            f_mant = (mant << 13);
        }
    } else if (exp == 31) {
        // 无穷大或NaN
        f_exp = 0xFF;
        f_mant = (mant == 0) ? 0 : 0x7FFFFF; // 如果尾数为0，则为无穷大，否则为NaN
    } else {
        // 正规数
        f_exp = exp - 15 + 127; // half偏置为15，float偏置为127
        f_mant = (mant << 13);
    }

    // 组合成float的位模式
    uint32_t f = (sign << 31) | (f_exp << 23) | f_mant;
    return *reinterpret_cast<float*>(&f);
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> offsetShape = {1};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scaleHostData = {2};
    std::vector<float> offsetHostData = {3};
    std::vector<short> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT8, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t dstType = 1;
    bool sqrtMode = true;

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAscendAntiQuant第一段接口
    ret = aclnnAscendAntiQuantGetWorkspaceSize(self, scale, offset, dstType, sqrtMode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendAntiQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAscendAntiQuant第二段接口
    ret = aclnnAscendAntiQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendAntiQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改，查看resultData中数据
    auto size = GetShapeSize(outShape);
    std::vector<short> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(short),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        // 以整数形式输出half的位模式
        LOG_PRINT("result[%ld] is: %hd (integer)\n", i, resultData[i]);

        // 或者，如果你想要输出浮点数值
        float val = half_to_float(static_cast<uint16_t>(resultData[i]));
        LOG_PRINT("result[%ld] is: %f (float)\n", i, val);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
