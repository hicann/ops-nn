
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EdgeSoftmaxTilingData)
TILING_DATA_FIELD_DEF(int32_t, E);
TILING_DATA_FIELD_DEF(int16_t, F);
TILING_DATA_FIELD_DEF(int16_t, N);
TILING_DATA_FIELD_DEF(int8_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EdgeSoftmax, EdgeSoftmaxTilingData)
}  // namespace optiling
