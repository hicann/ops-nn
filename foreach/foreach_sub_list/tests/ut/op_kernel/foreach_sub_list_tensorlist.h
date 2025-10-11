/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * limitations under the License.
 */

/*!
 * \file foreach_sub_list_tensorlist.h
 * \brief
 */

#ifndef FOREACH_SUB_LIST_TENSORLIST_H
#define FOREACH_SUB_LIST_TENSORLIST_H
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../data_utils.h"

template <typename T1, typename T2>
inline T1 CeilA2B(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

namespace ForeachSubList {
template <typename T>
uint8_t* CreateTensorList(const std::vector<std::vector<uint64_t>>& shapeInfos, char* d_type, int arg_pos)
{
    uint64_t tensorListDescCount = 1 + shapeInfos.size() * 2;
    for (auto s : shapeInfos) {
        tensorListDescCount += s.size();
    }
    std::vector<uint64_t> shapeSizeList;
    uint64_t* tensorListDesc = (uint64_t*)AscendC::GmAlloc(tensorListDescCount * sizeof(uint64_t));
    *tensorListDesc = (tensorListDescCount - shapeInfos.size()) * sizeof(uint64_t);
    uint64_t addrIndex = 0;
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        addrIndex++;
        uint16_t dimCount = shapeInfos[i].size();
        *(tensorListDesc + addrIndex) = ((uint64_t)(i) << 32) + dimCount;
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < dimCount; j++) {
            addrIndex++;
            *(tensorListDesc + addrIndex) = shapeInfos[i][j];
            shapeSize *= shapeInfos[i][j];
        }
        shapeSizeList.push_back(shapeSize);
    }
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        addrIndex++;
        uint64_t dataSize = shapeSizeList[i] * sizeof(T);
        uint8_t* dataPtr = (uint8_t*)AscendC::GmAlloc(CeilA2B(dataSize, 32) * 32);
        if (arg_pos != 0) {
            std::stringstream fileName;
            fileName << "./sub_list_data/" << d_type << "_input_t_foreach_sub" << i << "_" << arg_pos << ".bin";
            ReadFile(fileName.str(), dataSize, dataPtr, dataSize);
        }
        *(tensorListDesc + addrIndex) = (uint64_t)dataPtr;
    }
    return (uint8_t*)tensorListDesc;
}

template <typename T>
void FreeTensorList(uint8_t* addr, const std::vector<std::vector<uint64_t>>& shapeInfos, char* d_type, int arg_pos)
{
    uint64_t dataPtrOffset = *((uint64_t*)addr);
    uint8_t* dataAddr = addr + dataPtrOffset;
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < shapeInfos[i].size(); j++) {
            shapeSize *= shapeInfos[i][j];
        }
        uint8_t* tensorAddr = (uint8_t*)(*((uint64_t*)(dataAddr) + i));
        if (arg_pos == 0) {
            std::stringstream fileName;
            fileName << "./sub_list_data/" << d_type << "_output_t_foreach_sub" << i << ".bin";
            WriteFile(fileName.str(), tensorAddr, shapeSize * sizeof(T));
        }
        AscendC::GmFree((void*)(tensorAddr));
    }

    AscendC::GmFree((void*)addr);
}
} // namespace ForeachSubList

#endif // FOREACH_SUB_LIST_TENSORLIST_H
