#include "edge_softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int32_t DATA_BLOCK_SIZE = 32;  // 数据块大小: 32字节
constexpr int32_t DATA_BLOCK_LEN_32 =
    DATA_BLOCK_SIZE / sizeof(float);  // 32位数据块长度: 32字节/4字节=8个元素
constexpr int32_t REPEAT_SIZE = 256;  // 重复操作大小: 256字节
constexpr int32_t REPEAT_LEN_32 =
    REPEAT_SIZE / sizeof(float);  // 32位重复操作长度: 256字节/4字节=64个元素
constexpr int32_t coreNum = 40;

constexpr inline int32_t CeilDiv(int32_t a, int32_t b) {
    return a == 0 ? 0 : static_cast<int32_t>(1) + (a - 1) / b;
}
constexpr inline int32_t AlignUp(int32_t a, int32_t b) { return CeilDiv(a, b) * b; }
constexpr inline int32_t AlignDown(int32_t a, int32_t b) { return a / b * b; }
constexpr inline int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }

struct InputInfo {
    explicit InputInfo(gert::TilingContext* context) {
        shape = context->GetInputShape(0);
        const auto& storage = shape->GetStorageShape();
        auto attr = context->GetAttrs()->GetAttrPointer<int>(0);

        E = storage.GetDim(0);
        F = storage.GetDim(1);
        N = *attr;
    }

    const gert::StorageShape* shape = nullptr;
    int32_t E = 0;
    int32_t F = 0;
    int32_t N = 0;
};

namespace optiling {
static ge::graphStatus EdgeSoftmaxTilingFunc(gert::TilingContext* context) {
    // 获取输入信息
    InputInfo inputInfo{context};
    // 设置核数
    // int32_t blockNum = Min(inputInfo.N, coreNum);
    int32_t blockNum = (inputInfo.F % DATA_BLOCK_LEN_32 == 0) ? Min(inputInfo.N, coreNum) : 1;
    context->SetBlockDim(blockNum);

    // // 调试打印
    // std::cout << "Input Info: " << "\n"
    //           << "  E: " << inputInfo.E << "\n"
    //           << "  F: " << inputInfo.F << "\n"
    //           << "  N: " << inputInfo.N << "\n";

    // 设置tiling data
    EdgeSoftmaxTilingData tiling_data;
    tiling_data.set_E(inputInfo.E);
    tiling_data.set_F(static_cast<int16_t>(inputInfo.F));
    tiling_data.set_N(static_cast<int16_t>(inputInfo.N));
    tiling_data.set_blockNum(static_cast<int8_t>(blockNum));
    tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                             context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class EdgeSoftmax : public OpDef {
   public:
    explicit EdgeSoftmax(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("N").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::EdgeSoftmaxTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(EdgeSoftmax);
}  // namespace ops