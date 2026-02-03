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
 * \file main.cpp
 * \brief
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    int64_t m = 1;
    int64_t n = 12288;
    int64_t k = 8192;
    int64_t groupSize = 128;
    // define operator
    std::vector<int64_t> shapeA{m, k};
    std::vector<int64_t> shapeWeight{k, n};
    std::vector<int64_t> shapeAntiquantScale{k / groupSize, n};
    std::vector<int64_t> shapeC{m, n};
    aclDataType dataTypeA = ACL_FLOAT16;
    aclDataType dataTypeWeight = ACL_INT4;
    aclDataType dataTypeAntiquantScale = ACL_FLOAT16;
    aclDataType dataTypeC = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataTypeA, shapeA.size(), shapeA.data(), format);
    opDesc.AddInputTensorDesc(dataTypeWeight, shapeWeight.size(), shapeWeight.data(), format);
    opDesc.AddInputTensorDesc(dataTypeAntiquantScale, shapeAntiquantScale.size(), shapeAntiquantScale.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeC, shapeC.size(), shapeC.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_a.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_weight.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/input_antiquant_scale_fp16.bin", fileSize, runner.GetInputBuffer<void>(2),
             runner.GetInputSize(2));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("../output/output_z.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    return SUCCESS;
}
