#ifndef __MASKED_SOFTMAX_WITH_RELPOSBIAS_TILING_H__
#define __MASKED_SOFTMAX_WITH_RELPOSBIAS_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MaskedSoftmaxWithRelPosBiasTilingData {
  uint32_t tailStartCoreIdx;
  uint32_t singleCoreSize;
  uint32_t w;
  uint32_t n;
  uint32_t s1;
  uint32_t s2;
  uint32_t wns1s2;
  uint32_t wns1;
  uint32_t ws1s2;
  uint32_t ns1s2;
  uint32_t ws1;
  uint32_t ns1;
  uint32_t s1s2;
  uint32_t wns1s2Aligned;
  uint32_t s1s2Aligned;
  uint32_t ns1s2Aligned;
  uint32_t s2Aligned;
  uint32_t s2DtypeSize;
  uint32_t inQueueLen;
  uint32_t maskQueueLen;
  uint32_t attenBiasNum;
  uint32_t stackNum;
  uint32_t loopNum;
  uint32_t loopTailNum;
  uint32_t xCopyEleNum;
  uint32_t tailXCopyEleNum;
  float scaleValue;
  uint32_t castTempSize;
  uint32_t tmpXubSize;
  uint32_t tmpMaskUbSize;
  SoftMaxTiling softmaxTilingData;
  SoftMaxTiling tailSoftmaxTilingData;
};
#pragma pack()

inline void InitMaskedSoftmaxWithRelPosBiasTilingData(uint8_t* tiling,
                                                       MaskedSoftmaxWithRelPosBiasTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(MaskedSoftmaxWithRelPosBiasTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)      \
  MaskedSoftmaxWithRelPosBiasTilingData tiling_data; \
  InitMaskedSoftmaxWithRelPosBiasTilingData(tiling_arg, &tiling_data)
#endif
