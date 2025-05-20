#include "EvalNetwork.h"

#include <assert.h>
#include <fstream>
#include <iostream>

#include <optix.h>

#include "sutil/shared/structs.h"
#include <sutil/Buffers/BufferAlloc.h>

void Network::createNetworkBuffers(LaunchParams *lp, sutil::BufferAlloc &bufferalloc)
{
    if (networkPath.empty())
    {
        printf("\nNetwork Path Empty\n\n");
        return;
    }
    else
        printf("\nNetwork Path %s\n\n", networkPath.c_str());

    unsigned int input_size = infnetwork::input_size; // Embeded input size
    unsigned int output_size = infnetwork::output_size;
    unsigned int layer_size = infnetwork::layer_size;

    // First layer is 256x10
    lp->inf_network.w0 =
        createNetworkLayerBuffer(bufferalloc, input_size, layer_size, "w_0.txt");
    lp->inf_network.b0 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_0.txt");

    // Second layer is 256x256
    lp->inf_network.w1 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_1.txt");
    lp->inf_network.b1 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_1.txt");

    //->Third layer is 256x256
    lp->inf_network.w2 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_2.txt");
    lp->inf_network.b2 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_2.txt");

    //->Fourth layer 256x256
    lp->inf_network.w3 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_3.txt");
    lp->inf_network.b3 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_3.txt");

    //->Fifth layer is 256x256
    lp->inf_network.w4 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_4.txt");
    lp->inf_network.b4 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_4.txt");

    //->Sixth layer is 256x256
    lp->inf_network.w5 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_5.txt");
    lp->inf_network.b5 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_5.txt");

    //->Seventh layer is 256x256
    lp->inf_network.w6 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_6.txt");
    lp->inf_network.b6 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_6.txt");

    //->Eighth layer is 256x256
    lp->inf_network.w7 =
        createNetworkLayerBuffer(bufferalloc, layer_size, layer_size, "w_7.txt");
    lp->inf_network.b7 =
        createNetworkLayerBuffer(bufferalloc, 1, layer_size, "b_7.txt");

    //->Nineth layer is 3x256
    lp->inf_network.w8 =
        createNetworkLayerBuffer(bufferalloc, layer_size, output_size, "w_8.txt");
    lp->inf_network.b8 =
        createNetworkLayerBuffer(bufferalloc, 1, output_size, "b_8.txt");
    //   lp->inf_network.flux =
    //       createNetworkLayerBuffer(scene, 1, output_size, "flux.txt");

    //   lp->inf_network.beckmann_shininess =
    //       createNetworkLayerBuffer(scene, 1, 1, "beckmann.txt");

    //   lp->inf_network.normal = make_float3(0, 0, 1);
}

BufferView<float> Network::createNetworkLayerBuffer(sutil::BufferAlloc &bufferalloc,
                                                    unsigned int input_n,
                                                    unsigned int output_n,
                                                    const std::string &file)
{
    std::string filename = networkPath + "/" + file;
    float *buffer = loadArrayFromFile(filename, input_n * output_n);
    assert(buffer);

    BufferView<float> buffer_view;
    unsigned int idx = bufferalloc.addBuffer(input_n * output_n * sizeof(float),
                                             reinterpret_cast<const void *>(buffer));
    buffer_view.data = bufferalloc.getBuffer(idx);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(input_n * output_n);
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
    return buffer_view;
}

float *Network::loadArrayFromFile(std::string &filename, unsigned int size)
{
    std::ifstream inputfile(filename);

    if (!inputfile.is_open())
    {
        printf("error opening file %s\n", filename.c_str());
        return nullptr;
    }

    float *arr = (float *)malloc(sizeof(float) * size);
    for (unsigned int i = 0; i < size; i++)
    {
        inputfile >> arr[i];
    }
    return arr;
}