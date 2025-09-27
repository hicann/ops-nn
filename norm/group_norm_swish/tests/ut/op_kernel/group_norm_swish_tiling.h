#ifndef _GE_GROUP_NORM_SWISH_TILING_H_
#define _GE_GROUP_NORM_SWISH_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DTYPE_X half
#define DTYPE_GAMMA float
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#pragma pack(1)

struct GroupNormSwishTilingData {
  int64_t numGroups;
  float epsilon;
  int64_t activateSwish;
  float swishScale;
  int64_t hwNum;
  int64_t shapeC;
  int64_t shapeCAlign;
  int64_t shapeD;
  int64_t numPerGroup;
  int64_t groupPerCore;
  int64_t groupLastCore;
  int64_t groupPerCoreAlign;
  int64_t numPerLoop;
  int64_t loopTimes;
  int64_t loopTimesAlign;
  int64_t numTailLoop;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
  GroupNormSwishTilingData tilingData;                                          \
  INIT_TILING_DATA(GroupNormSwishTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).numGroups = tilingDataPointer->numGroups;                 \
  (tilingData).epsilon = tilingDataPointer->epsilon;                         \
  (tilingData).activateSwish = tilingDataPointer->activateSwish;                     \
  (tilingData).swishScale = tilingDataPointer->swishScale;                       \
  (tilingData).hwNum = tilingDataPointer->hwNum;                       \
  (tilingData).shapeC = tilingDataPointer->shapeC;             \
  (tilingData).shapeCAlign = tilingDataPointer->shapeCAlign;               \
  (tilingData).shapeD = tilingDataPointer->shapeD;             \
  (tilingData).numPerGroup = tilingDataPointer->numPerGroup;             \
  (tilingData).groupPerCore = tilingDataPointer->groupPerCore;                     \
  (tilingData).groupLastCore = tilingDataPointer->groupLastCore;                   \
  (tilingData).groupPerCoreAlign = tilingDataPointer->groupPerCoreAlign;           \
  (tilingData).numPerLoop = tilingDataPointer->numPerLoop;         \
  (tilingData).loopTimes = tilingDataPointer->loopTimes;                 \
  (tilingData).loopTimesAlign = tilingDataPointer->loopTimesAlign;                 \
  (tilingData).numTailLoop = tilingDataPointer->numTailLoop;

#endif
