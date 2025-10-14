#ifndef _GE_GROUP_NORM_SILU_TILING_H_
#define _GE_GROUP_NORM_SILU_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DTYPE_X half
#define DTYPE_GAMMA float
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#pragma pack(1)

// struct GroupNormSiluTilingData {
//   int64_t numGroups;
//   int64_t hwNum;
//   int64_t elemNum;
//   int64_t shapeC;
//   int64_t shapeD;
//   int64_t realCoreNum;
//   int64_t numPerCore;
//   int64_t numLastCore;
//   int64_t processSize;
//   int64_t loopNum;
//   int64_t loopTail;
//   int64_t innerLoopNum;
//   int64_t innerLoopTail;
//   int64_t tilingKey;
//   float epsilon;
//   int64_t activateSilu;
// };

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
  GroupNormSiluTilingData tilingData;                                          \
  INIT_TILING_DATA(GroupNormSiluTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).numGroups = tilingDataPointer->numGroups;                 \
  (tilingData).hwNum = tilingDataPointer->hwNum;                         \
  (tilingData).elemNum = tilingDataPointer->elemNum;                     \
  (tilingData).shapeC = tilingDataPointer->shapeC;                       \
  (tilingData).shapeD = tilingDataPointer->shapeD;                       \
  (tilingData).realCoreNum = tilingDataPointer->realCoreNum;             \
  (tilingData).numPerCore = tilingDataPointer->numPerCore;               \
  (tilingData).numLastCore = tilingDataPointer->numLastCore;             \
  (tilingData).processSize = tilingDataPointer->processSize;             \
  (tilingData).loopNum = tilingDataPointer->loopNum;                     \
  (tilingData).loopTail = tilingDataPointer->loopTail;                   \
  (tilingData).innerLoopNum = tilingDataPointer->innerLoopNum;           \
  (tilingData).innerLoopTail = tilingDataPointer->innerLoopTail;         \
  (tilingData).tilingKey = tilingDataPointer->tilingKey;                 \
  (tilingData).epsilon = tilingDataPointer->epsilon;                 \
  (tilingData).activateSilu = tilingDataPointer->activateSilu;

#endif
