/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "adump_api.h"
#include "acl/acl_base.h"
#include "register/op_info_record_registry.h"

namespace Adx {
bool AdumpIsDumpEnable(DumpType dumpType) { return false;}
int32_t AdumpSetDumpConfig(DumpType dumpType, const DumpConfig &dumpConfig){return 0;}
int32_t AdumpDumpTensor(const std::string &opType, const std::string &opName,
                        const std::vector<TensorInfo> &tensors, aclrtStream stream) {
    return 0;
}

int32_t AdumpAddExceptionOperatorInfo(const OperatorInfo &opInfo) {
  return 0;
}

void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex) {
  return nullptr;
}

void *AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t &atomicIndex) {
  return nullptr;
}

void *AdumpGetDFXInfoAddrForStatic(uint32_t space, uint64_t &atomicIndex) {
  return nullptr;
}

void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize, 
                         aclrtStream stream, const char *opType){
  return;
}

void AdumpPrintAndGetTimeStampInfo(const void *workSpaceAddr, const size_t dumpWorkSpaceSize, 
                         aclrtStream stream, const char *opType, std::vector<MsprofAicTimeStampInfo> & timeStampInfo ){
  return;
}

void AdumpPrintSetConfig(const AdumpPrintConfig &config){
  return;
}

int32_t AdumpDelExceptionOperatorInfo(uint32_t deviceId, uint32_t streamId) {
  return 0;
}

}

void OpInfoRecord::OpInfoRecordRegister::ExeOptInfoStat(
  const gert::TilingContext *ctx,
  const OpInfoRecord::OpCompilerOption &opt,
  const OpInfoRecord::OpKernelInfo *kernelInfo) const
{
  return;
}

OpInfoRecord::OpInfoRecordRegister *OpInfoRecord::OpInfoRecordRegister::Instance()
{
  static OpInfoRecord::OpInfoRecordRegister instance;
  return &instance;
}

void OpInfoRecord::OpInfoRecordRegister::RegNotify(
  const OpInfoRecord::OpInfoRecordRegister::NotifyFn notifyFn) const
{
  return;
}

bool OpInfoRecord::OpInfoRecordRegister::GetSwitchState() const
{
  return false;
}