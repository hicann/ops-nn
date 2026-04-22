/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define protected public
#define private public

#include "cube_utils/cube_addinputstrategy.h"
#include "cube_utils/cube_common.h"

#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace ops;

namespace ops {
const std::string GRAPH_NAME = "test_graph";

class cube_addinputstrategy_ut : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cube_addinputstrategy_ut SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "cube_addinputstrategy_ut TearDown" << std::endl;
    }

    static void DumpTestGraph(ge::Graph &graph) {
        (void)graph;
        std::cout << "cube_test : dump graph" << std::endl;
    }

    static void CreateEdge(ge::Graph &graph, const ge::GNodePtr &src, int src_idx,
                          const ge::GNodePtr &dst, int dst_idx) {
        auto ret = graph.AddDataEdge(*src, src_idx, *dst, dst_idx);
        if (ret != ge::GRAPH_SUCCESS) {
            std::cout << "cube_test : AddDataEdge failed" << std::endl;
        }
    }

    static ge::GNodePtr CreateConstNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
        tensor_desc.SetOriginShape(shape);

        std::vector<float> data;
        for (int i = 0; i != 3; i++) {
            for (int j = 0; j != 4; j++) {
                for (int k = 0; k != 5; k++) {
                    for (int m = 0; m != 6; m++) {
                        float dst = i * j * k * m;
                        data.push_back(dst);
                    }
                }
            }
        }
        std::cout << "cube_test inputv01 size = " << data.size() << std::endl;

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Const")
            .Name(opname.c_str())
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateOutputDesc(0, tensor_desc);
        ge::Tensor weight_tensor(tensor_desc,
            reinterpret_cast<const uint8_t*>(data.data()),
            data.size() * sizeof(float));
        node.SetAttr(kAscendNameWeightAsc, weight_tensor);

        ge::GNodePtr node_ptr = std::make_shared<ge::GNode>(node);
        return node_ptr;
    }

    static ge::GNodePtr CreatePostCubeNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("FixPipe")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x0", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x3", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x4", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x5", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x6", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x7", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x8", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x9", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        for (size_t i = 0; i < 10; ++i) {
            node.UpdateInputDesc(i, desc);
        }
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateDequantNode(ge::Graph &graph, const std::string &opname,
                                          bool hasattr = false,
                                          float scale = 0.0f, float offset = 0.0f) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto const_node_ptr = CreateConstNode(graph, opname + "_constinput");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendDequant")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"deq_scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        if (hasattr) {
            node.SetAttr(kAscendScaleAsc, scale);
            node.SetAttr(kAscendOffsetAsc, offset);
        }

        CreateEdge(graph, const_node_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateQuantNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendQuant")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        float offset = 8.0f;
        float scale_val = 4.0f;
        node.SetAttr(kAscendOffsetAsc, offset);
        node.SetAttr(kAscendScaleAsc, scale_val);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateRequantNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendRequant")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        auto const_node_ptr = CreateConstNode(graph, opname + "_constinput");
        CreateEdge(graph, const_node_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateCastNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Cast")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateAntiQuantNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendAntiQuant")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        float offset = 8.0f;
        float scale_val = 4.0f;
        node.SetAttr(kAscendOffsetAsc, offset);
        node.SetAttr(kAscendScaleAsc, scale_val);

        auto const_node_ptr = CreateConstNode(graph, opname + "_inputconstnode");
        CreateEdge(graph, const_node_ptr, 0, std::make_shared<ge::GNode>(node), 0);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateAddNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Add")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        auto inputnode_ptr = CreateAntiQuantNode(graph, opname + "_inputantinode");
        CreateEdge(graph, inputnode_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateAdd2Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Add")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        auto constnode_ptr = CreateConstNode(graph, opname + "_inputconstnode");
        CreateEdge(graph, constnode_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateAdd3Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
        desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Add")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateAdd4Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
        desc.SetOriginFormat(ge::FORMAT_ND);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Add")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateSubNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Sub")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        auto inputnode_ptr = CreateAntiQuantNode(graph, opname + "_inputantinode");
        CreateEdge(graph, inputnode_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateReluNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Relu")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreatePReluNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto const_node_ptr = CreateConstNode(graph, opname + "_inputconstnode");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("PRelu")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"slope", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        CreateEdge(graph, const_node_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreatePRelu2Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

        auto const_node_ptr = CreateConstNode(graph, opname + "_inputconstnode");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("PRelu")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"slope", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        CreateEdge(graph, const_node_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreatePRelu3Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("PRelu")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"slope", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateTanhNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto const_node_ptr = CreateConstNode(graph, opname + "_inputconstnode");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Tanh")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        CreateEdge(graph, const_node_ptr, 0, std::make_shared<ge::GNode>(node), 1);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateLReluNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("LeakyRelu")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        float negative_slope = 8.0f;
        node.SetAttr(kAscendNegativeSlopeAsc, negative_slope);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateRelu6Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Relu6")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateTransDataNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape_in({3, 4, 5, 6});
        ge::Shape shape_out({3, 4, 5, 6});
        ge::TensorDesc desc_in(shape_in, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
        ge::TensorDesc desc_out(shape_out, ge::FORMAT_ND, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("TransData")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc_in);
        node.UpdateOutputDesc(0, desc_out);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateConv2DNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Conv2d")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"filter", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateOtherNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("MaxPool")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    static ge::GNodePtr CreateNodeSimpleFactorTemplate(ge::Graph &graph, const std::string &opname,
                                                        const std::string &opunittype) {
        if (opunittype == "A") {
            return CreateConv2DNode(graph, opname);
        } else if (opunittype == "B") {
            return CreateDequantNode(graph, opname);
        } else if (opunittype == "C") {
            return CreateReluNode(graph, opname);
        } else if (opunittype == "D") {
            return CreateAddNode(graph, opname);
        } else if (opunittype == "E") {
            return CreateQuantNode(graph, opname);
        } else if (opunittype == "F") {
            return CreateTransDataNode(graph, opname);
        } else {
            return CreateOtherNode(graph, opname);
        }
    }
};

// ==================== IsScalar Tests ====================

TEST_F(cube_addinputstrategy_ut, IsScalar01) {
    PostCubeAddInputBase m_testpass{};
    ge::Shape shape{};
    bool ret = m_testpass.IsScalar(shape);
    EXPECT_EQ(ret, true);
    ge::Shape original_shape = ge::Shape({128, 12, 5, 6, 16});
    ret = m_testpass.IsScalar(original_shape);
    EXPECT_EQ(ret, false);
}

// ==================== CreateAndUpdateVectorMulsInput Tests ====================

TEST_F(cube_addinputstrategy_ut, CreateAndUpdateVectorMulsInput01) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.CreateAndUpdateVectorMulsInput(graph, funtcparam, dequant_node_info, prelu_node_info, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// ==================== CloneVectorInput Tests ====================

TEST_F(cube_addinputstrategy_ut, CloneVectorInput01) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.CloneVectorInput(graph, prelu_node_info, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, CloneVectorInput02) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePRelu2Node(graph, "prelu");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.CloneVectorInput(graph, prelu_node_info, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, CloneVectorInput03) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePRelu3Node(graph, "prelu");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.CloneVectorInput(graph, prelu_node_info, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cube_addinputstrategy_ut, CloneVectorInput04) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePRelu2Node(graph, "prelu");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.CloneVectorInput(graph, prelu_node_info, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// ==================== CreateAndRelinkCastNode Tests ====================

TEST_F(cube_addinputstrategy_ut, CreateAndRelinkCastNode01) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());
    ge::GNodePtr inputnode = nullptr;
    ge::GNodePtr outputnode = nullptr;
    int index = 3;
    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.CreateAndRelinkCastNode(graph, inputnode, outputnode, index, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// ==================== SetClipValue6 Tests ====================

TEST_F(cube_addinputstrategy_ut, SetClipValue601) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);

    std::vector<ge::GNodePtr> new_nodes;
    bool ret = true;
    m_testpass.SetClipValue6(graph, funtcparam, ge::DT_FLOAT, new_nodes);
    m_testpass.SetClipValue6(graph, funtcparam, ge::DT_FLOAT16, new_nodes);
    m_testpass.SetClipValue6(graph, funtcparam, ge::DT_INT8, new_nodes);
    m_testpass.SetClipValue6(graph, funtcparam, ge::DT_INT4, new_nodes);
    m_testpass.SetClipValue6(graph, funtcparam, ge::DT_INT32, new_nodes);
    EXPECT_EQ(ret, true);
}

// ==================== Strategy DoAddInput Tests ====================

TEST_F(cube_addinputstrategy_ut, DoAddInput22) {
    PostCubeAddInputStrategy22 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    PostCubeNodeInfo dequant_node_info(dequant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    PostCubeFunctionParamPtr funtcparam;
    uint32_t index = 0;
    funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dequant + relu + quant
TEST_F(cube_addinputstrategy_ut, DoAddInput23) {
    PostCubeAddInputStrategy24 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto relu_node = CreateReluNode(graph, "relu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo relu_node_info(relu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(relu_node_info);
    match_pass.m_opnodes.push_back(quant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput24) {
    PostCubeAddInputStrategy24 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto leaky_relu_node = CreateLReluNode(graph, "lrelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo lrelu_node_info(leaky_relu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(lrelu_node_info);
    match_pass.m_opnodes.push_back(quant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput25) {
    PostCubeAddInputStrategy25 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(prelu_node_info);
    match_pass.m_opnodes.push_back(quant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 0;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput25_001) {
    PostCubeAddInputStrategy25 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    CreateEdge(graph, dequant_node, 0, prelu_node, 0);
    CreateEdge(graph, prelu_node, 0, quant_node, 0);

    ge::Shape shape1({1, -1, 14, 14});
    ge::Shape shape2({1, -1, 14, 14});

    ge::TensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginShape(shape1);

    ge::TensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginShape(shape2);

    prelu_node->UpdateInputDesc(1, out_desc1);
    quant_node->UpdateInputDesc(1, out_desc1);

    ge::Format format1 = ge::FORMAT_NC1HWC0;
    prelu_node->UpdateInputDesc(1, out_desc1);
    quant_node->UpdateInputDesc(1, out_desc2);

    PostCubeAddInputStrategy25::UpdatePostNodeShape(*prelu_node, *quant_node);
    ge::TensorDesc quant_input_desc;
    (void)quant_node->GetInputDesc(1, quant_input_desc);
    EXPECT_EQ(quant_input_desc.GetShape().GetDims(), shape1.GetDims());
}

TEST_F(cube_addinputstrategy_ut, DoAddInput25_002) {
    PostCubeAddInputStrategy25 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    CreateEdge(graph, dequant_node, 0, prelu_node, 0);
    CreateEdge(graph, prelu_node, 0, quant_node, 0);

    ge::Shape shape1({1, -1, 16, 14});
    ge::Shape shape2({1, -1, 14, 8});

    ge::TensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginShape(shape1);

    ge::TensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginShape(shape2);

    prelu_node->UpdateInputDesc(1, out_desc1);
    quant_node->UpdateInputDesc(1, out_desc2);

    PostCubeAddInputStrategy25::UpdatePostNodeShape(*prelu_node, *quant_node);
    ge::TensorDesc quant_input_desc;
    (void)quant_node->GetInputDesc(1, quant_input_desc);
    EXPECT_EQ(quant_input_desc.GetShape().GetDims(), shape2.GetDims());
}

TEST_F(cube_addinputstrategy_ut, DoAddInput25_003) {
    PostCubeAddInputStrategy25 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    CreateEdge(graph, dequant_node, 0, prelu_node, 0);
    CreateEdge(graph, prelu_node, 0, quant_node, 0);

    ge::Shape shape1({1, 1, 1, 1});
    ge::Shape shape2({1, -1, 14, 8});

    ge::TensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginShape(shape1);

    ge::TensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginShape(shape2);

    prelu_node->UpdateInputDesc(1, out_desc1);
    quant_node->UpdateInputDesc(1, out_desc2);

    PostCubeAddInputStrategy25::UpdatePostNodeShape(*prelu_node, *quant_node);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput25_004) {
    PostCubeAddInputStrategy25 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    ge::Shape ori_shape1({36});
    ge::Shape ori_shape2({1, 36, 407});
    ge::Shape shape1({1, 3, 1, 1, 16});
    ge::Shape shape2({1, 3, 1, 407, 16});

    ge::TensorDesc tensor_desc(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    ge::TensorDesc tensor_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc.SetOriginShape(ori_shape1);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginShape(ori_shape2);

    ge::GNode setquantscale_node = ge::es::CompliantNodeBuilder(&graph)
        .OpType("SetQuantScale")
        .Name("setquantscale")
        .IrDefInputs({
            {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();

    setquantscale_node.UpdateOutputDesc(0, tensor_desc);
    setquantscale_node.UpdateInputDesc(0, tensor_desc);
    setquantscale_node.UpdateInputDesc(1, tensor_desc2);

    ge::GNodePtr setquantscale_node_ptr = std::make_shared<ge::GNode>(setquantscale_node);

    PostCubeAddInputStrategy25::UpdateQuantScaleNodeShape(*setquantscale_node_ptr);
    ge::TensorDesc setquantscale_input_desc;
    (void)setquantscale_node_ptr->GetInputDesc(0, setquantscale_input_desc);
    EXPECT_EQ(setquantscale_input_desc.GetShape().GetDims(), shape1.GetDims());
}

TEST_F(cube_addinputstrategy_ut, DoAddInput31) {
    PostCubeAddInputStrategy31 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(quant_node_info);
    match_pass.m_opnodes.push_back(prelu_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput32) {
    PostCubeAddInputStrategy32 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(prelu_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput33) {
    PostCubeAddInputStrategy33 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(prelu_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(0);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInputStrategy52) {
    PostCubeAddInputStrategy52 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto tanh_node = CreateTanhNode(graph, "tanh");
    auto quant_node = CreateQuantNode(graph, "quant");

    CreateEdge(graph, dequant_node, 0, tanh_node, 0);
    CreateEdge(graph, tanh_node, 0, quant_node, 0);

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo tanh_node_info(tanh_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(tanh_node_info);
    match_pass.m_opnodes.push_back(quant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(2);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput61) {
    PostCubeAddInputStrategy61 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(prelu_node_info);
    match_pass.m_opnodes.push_back(quant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput43) {
    PostCubeAddInputStrategy43 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant", true, (float)66.55, (float)128.0);
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");
    auto relu6_node = CreateRelu6Node(graph, "relu6");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);
    PostCubeNodeInfo relu6_node_info(relu6_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(relu6_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    dequant_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT));
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    dequant_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    dequant_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_INT8));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    dequant_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_INT4));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    dequant_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_INT32));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput42) {
    PostCubeAddInputStrategy42 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto quant_node = CreateQuantNode(graph, "quant");
    auto relu6_node = CreateRelu6Node(graph, "relu6");

    PostCubeNodeInfo quant_node_info(quant_node);
    PostCubeNodeInfo relu6_node_info(relu6_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(quant_node_info);
    match_pass.m_opnodes.push_back(relu6_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 4;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    relu6_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT));
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    relu6_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    relu6_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_INT8));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    relu6_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_INT4));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    relu6_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_INT32));
    ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput62) {
    PostCubeAddInputStrategy62 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(prelu_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(0);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput91) {
    PostCubeAddInputStrategy91 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto quant_node = CreateQuantNode(graph, "quant");
    PostCubeNodeInfo quant_node_info(quant_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(quant_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_INT8);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(0);

    std::vector<ge::GNodePtr> new_nodes;
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// ==================== UpdateVectorMulsOutputTensorDesc Tests ====================

TEST_F(cube_addinputstrategy_ut, UpdateVectorMulsOutputTensorDesc01) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    ge::TensorDesc dequant_weight_inputdesc;
    ge::TensorDesc prelu_weight_inputdesc;
    (void)dequant_node->GetInputDesc(1, dequant_weight_inputdesc);
    (void)prelu_node->GetInputDesc(1, prelu_weight_inputdesc);
    dequant_weight_inputdesc.SetShape(ge::Shape({3, 4, 5, 6}));
    dequant_weight_inputdesc.SetOriginShape(ge::Shape({3, 4, 5, 6}));

    prelu_weight_inputdesc.SetShape(ge::Shape({3, 4, 5, 6}));
    prelu_weight_inputdesc.SetOriginShape(ge::Shape({3, 4, 5, 6}));

    ge::TensorDesc out_tensor_desc = dequant_weight_inputdesc;
    auto ret = m_testpass.UpdateVectorMulsOutputTensorDesc(dequant_weight_inputdesc, prelu_weight_inputdesc,
                                                           out_tensor_desc);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, UpdateVectorMulsOutputTensorDesc02) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    ge::TensorDesc dequant_weight_inputdesc;
    ge::TensorDesc prelu_weight_inputdesc;
    (void)dequant_node->GetInputDesc(1, dequant_weight_inputdesc);
    (void)prelu_node->GetInputDesc(1, prelu_weight_inputdesc);
    dequant_weight_inputdesc.SetShape(ge::Shape({1, 1, 16, 1, 16}));
    dequant_weight_inputdesc.SetOriginShape(ge::Shape({256}));

    prelu_weight_inputdesc.SetShape(ge::Shape({1, 16, 1, 1, 16}));
    prelu_weight_inputdesc.SetOriginShape(ge::Shape({256}));

    ge::TensorDesc out_tensor_desc = dequant_weight_inputdesc;
    auto ret = m_testpass.UpdateVectorMulsOutputTensorDesc(dequant_weight_inputdesc, prelu_weight_inputdesc,
                                                           out_tensor_desc);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, UpdateVectorMulsOutputTensorDesc03) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    ge::TensorDesc dequant_weight_inputdesc;
    ge::TensorDesc prelu_weight_inputdesc;
    (void)dequant_node->GetInputDesc(1, dequant_weight_inputdesc);
    (void)prelu_node->GetInputDesc(1, prelu_weight_inputdesc);
    dequant_weight_inputdesc.SetShape(ge::Shape({1}));
    dequant_weight_inputdesc.SetOriginShape(ge::Shape({1}));

    prelu_weight_inputdesc.SetShape(ge::Shape({1, 16, 1, 1, 16}));
    prelu_weight_inputdesc.SetOriginShape(ge::Shape({256}));

    ge::TensorDesc out_tensor_desc = dequant_weight_inputdesc;
    auto ret = m_testpass.UpdateVectorMulsOutputTensorDesc(dequant_weight_inputdesc, prelu_weight_inputdesc,
                                                           out_tensor_desc);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, UpdateVectorMulsOutputTensorDesc04) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    ge::TensorDesc dequant_weight_inputdesc;
    ge::TensorDesc prelu_weight_inputdesc;
    (void)dequant_node->GetInputDesc(1, dequant_weight_inputdesc);
    (void)prelu_node->GetInputDesc(1, prelu_weight_inputdesc);
    dequant_weight_inputdesc.SetShape(ge::Shape({1, 1, 16, 1, 16}));
    dequant_weight_inputdesc.SetOriginShape(ge::Shape({256}));

    prelu_weight_inputdesc.SetShape(ge::Shape({1}));
    prelu_weight_inputdesc.SetOriginShape(ge::Shape({1}));

    ge::TensorDesc out_tensor_desc = dequant_weight_inputdesc;
    auto ret = m_testpass.UpdateVectorMulsOutputTensorDesc(dequant_weight_inputdesc, prelu_weight_inputdesc,
                                                           out_tensor_desc);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, UpdateVectorMulsOutputTensorDesc05) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    ge::TensorDesc dequant_weight_inputdesc;
    ge::TensorDesc prelu_weight_inputdesc;
    (void)dequant_node->GetInputDesc(1, dequant_weight_inputdesc);
    (void)prelu_node->GetInputDesc(1, prelu_weight_inputdesc);
    dequant_weight_inputdesc.SetShape(ge::Shape({1, 16, 1, 1, 16}));
    dequant_weight_inputdesc.SetOriginShape(ge::Shape({256}));

    prelu_weight_inputdesc.SetShape(ge::Shape({1, 16, 56, 56, 16}));
    prelu_weight_inputdesc.SetOriginShape(ge::Shape({56, 56, 256}));

    ge::TensorDesc out_tensor_desc = dequant_weight_inputdesc;
    auto ret = m_testpass.UpdateVectorMulsOutputTensorDesc(dequant_weight_inputdesc, prelu_weight_inputdesc,
                                                           out_tensor_desc);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(cube_addinputstrategy_ut, UpdateVectorMulsOutputTensorDesc06) {
    PostCubeAddInputBase m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant");
    auto prelu_node = CreatePReluNode(graph, "prelu");

    ge::TensorDesc dequant_weight_inputdesc;
    ge::TensorDesc prelu_weight_inputdesc;
    (void)dequant_node->GetInputDesc(1, dequant_weight_inputdesc);
    (void)prelu_node->GetInputDesc(1, prelu_weight_inputdesc);
    dequant_weight_inputdesc.SetShape(ge::Shape({1, 16, 1, 1, 16}));
    dequant_weight_inputdesc.SetOriginShape(ge::Shape({256}));

    prelu_weight_inputdesc.SetShape(ge::Shape({1, 16, 56, 56, 17}));
    prelu_weight_inputdesc.SetOriginShape(ge::Shape({56, 56, 272}));

    ge::TensorDesc out_tensor_desc = dequant_weight_inputdesc;
    auto ret = m_testpass.UpdateVectorMulsOutputTensorDesc(dequant_weight_inputdesc, prelu_weight_inputdesc,
                                                           out_tensor_desc);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cube_addinputstrategy_ut, DoAddInput43_1) {
    PostCubeAddInputStrategy43 m_testpass{};
    ge::Graph graph(GRAPH_NAME.c_str());

    auto dequant_node = CreateDequantNode(graph, "dequant", true, 8.0f, 4.0f);
    auto prelu_node = CreatePReluNode(graph, "prelu");
    auto quant_node = CreateQuantNode(graph, "quant");
    auto relu6_node = CreateRelu6Node(graph, "relu6");

    PostCubeNodeInfo dequant_node_info(dequant_node);
    PostCubeNodeInfo prelu_node_info(prelu_node);
    PostCubeNodeInfo quant_node_info(quant_node);
    PostCubeNodeInfo relu6_node_info(relu6_node);

    PostCubePassInfo match_pass;
    match_pass.pass_index = 2;
    match_pass.m_flag = 1;
    match_pass.unit_index = 2;
    match_pass.m_opnodes.push_back(dequant_node_info);
    match_pass.m_opnodes.push_back(relu6_node_info);

    auto post_cube_node = CreatePostCubeNode(graph, "post_cube");
    uint32_t index = 3;
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_node, index);
    funtcparam->SetDataType(ge::DT_FLOAT);
    funtcparam->SetFirstIndex(0);
    funtcparam->SetSecondIndex(1);

    std::vector<ge::GNodePtr> new_nodes;
    dequant_node->UpdateOutputDesc(0, ge::TensorDesc(ge::Shape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT));
    auto ret = m_testpass.DoAddInput(graph, match_pass, funtcparam, new_nodes);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

}  // namespace ops
