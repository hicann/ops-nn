#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import functools
import random
import time
import multiprocessing as mp
import gc
import psutil

def calRelativediff(x,y,diffThd):
    if float(max(abs(x), abs(y))) < float((1.0/(1<<14))/diffThd):
        result = abs(float(x) - float(y))/(float((1.0/(1<<14))/diffThd) + 10e-10)
    else:
        result = abs(float(x)-float(y))/(float(max(abs(x),abs(y)))+10e-10)

    return result

def calRelativediffNumpy(dataCheck, dataExpect, diffThd):
    a = np.abs(np.subtract(dataCheck, dataExpect))
    b1 = np.maximum(np.abs(dataCheck), (np.abs(dataExpect)))
    b2 = float((1.0 / (1 << 14)) / diffThd)
    b = np.maximum(b1, b2)
    result = np.divide(a, np.add(b, 10e-10))
    return result

def displayOutput(dr, de, start, end,diffThd):

    print ('\n--------------------------------------------------------------------')
    print ('Loop \t  ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print ('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)

    if splitCount <= 20:
        for i in range(splitCount+1):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
    else:
        for i in range(10):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
        print ('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(splitCount-10+1,splitCount+1):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))

def displayErrorOutput(dr, de, rdiff ,start, end, diffThd):
    print ('\n----------------------------Error Line------------------------------')
    print ('--------------------------------------------------------------------')
    print ('Loop \t  ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print ('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)
    count = 0
    for i in range(len(rdiff)):
        j = i + start
        if rdiff[j] > diffThd:
            count += 1
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
        if count == 10:
            break

def dataCompare(dataCheck, dataExpect,  diffThd=0.01, pctThd=0.05):
    npu_shape=dataCheck.shape
    expect_shape=dataExpect.shape
    print('npu_out_shape:',npu_shape)
    print('expect_out_shape:',expect_shape)
    dataCheck=dataCheck.flatten()
    dataExpect=dataExpect.flatten()
    start=0
    end=dataCheck.size - 1
    rdiff1 = calRelativediffNumpy(dataCheck[:len(dataCheck)//2].astype(np.float32), dataExpect[:len(dataExpect)//2].astype(np.float32), diffThd)
    rdiff2= calRelativediffNumpy(dataCheck[len(dataCheck)//2:].astype(np.float32), dataExpect[len(dataExpect)//2:].astype(np.float32), diffThd)
    rdiff = np.concatenate((rdiff1,rdiff2),axis=0)
    print(len(rdiff1), len(rdiff2), len(rdiff))
    split_rdiff = rdiff[start:end+1]
    splitCount = int(end-start+1) if end != start else 1
    print('splitCount: ',  splitCount)
    print('float(splitCount): ',  float(splitCount))
    ltNum = rdiff[rdiff < diffThd].size
    j=0
    ltNum = ltNum + dataExpect[np.isinf(dataExpect)].size + dataExpect[np.isnan(dataExpect)].size
    ltPct = float(ltNum)/float(splitCount) * 100.0
    displayOutput(dataCheck, dataExpect, start, end,diffThd)
    pctThd = (1-pctThd) * 100.0
    result = "Pass" if (ltPct>=pctThd) else "Failed"
    if npu_shape != expect_shape:
        result = "Failed"

    print ('\n--------------------------------------------------------------------')
    print ('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print ('--------------------------------------------------------------------')
    print ('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd,pctThd,ltPct,result))

    if npu_shape != expect_shape:
        print('============ out_shape is not equal expect!')
    else:
        print('============ out_shape is equal expect!')

    if result == "Failed":
        displayErrorOutput(dataCheck, dataExpect, rdiff ,start, end, diffThd)
    return result, ltPct
