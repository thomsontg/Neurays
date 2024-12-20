#pragma once

#include <iostream>
#include <random>
#include <sutil/math/vec_math.h>
// #include "structs.h"

#include <torch/torch.h>
#include <torch/script.h>

struct Net : torch::nn::Module
{
  Net()
  {
    fc1 = register_module("fc1", torch::nn::Linear(10, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 256));
    fc3 = register_module("fc3", torch::nn::Linear(256, 256));
    fc4 = register_module("fc4", torch::nn::Linear(256, 256));
    fc5 = register_module("fc5", torch::nn::Linear(256, 256));
    fc6 = register_module("fc6", torch::nn::Linear(256, 256));
    fc7 = register_module("fc7", torch::nn::Linear(256, 256));
    fc8 = register_module("fc8", torch::nn::Linear(256, 256));
    fc9 = register_module("fc9", torch::nn::Linear(256, 3));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::leaky_relu(fc1->forward(x), 0.01);
    x = torch::leaky_relu(fc2->forward(x), 0.01);
    x = torch::leaky_relu(fc3->forward(x), 0.01);
    x = torch::relu(fc4->forward(x));
    x = torch::relu(fc5->forward(x));
    x = torch::relu(fc6->forward(x));
    x = torch::relu(fc7->forward(x));
    x = torch::relu(fc8->forward(x));
    x = torch::relu(fc9->forward(x));
    x = torch::sigmoid(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr}, fc6{nullptr}, fc7{nullptr}, fc8{nullptr}, fc9{nullptr};
};

class NetworkData
{
public:
  static float3 *m_x;
  static float2 *m_ray_vec;
  static float *m_hit_data;
  static float3 m_light_vec;
  static float3 *m_result_buffer;
  static int batchsize;
  static int buffersize;

  static std::vector<int> idx;

  torch::Device device;
  std::shared_ptr<Net> network;

  bool training = false;

  NetworkData() : device(torch::kCPU), network(std::make_shared<Net>()) {}

  void update_data(float3 *x, float2 *ray_vec, float *hit_data, float3 light_vec, float3 *result_buffer, int bufsize, int bsize = 20000)
  {
    m_x = x;
    m_ray_vec = ray_vec;
    m_hit_data = hit_data;
    m_light_vec = light_vec;
    m_result_buffer = result_buffer;
    buffersize = bufsize;
    batchsize = bsize;

    idx.clear();

    for (int i = 0; i < buffersize; i++)
    {
      if (m_hit_data[i] != 0.0f)
      {
        idx.push_back(i);
      }
    }

    std::cout << "USEFUL DATABUFFER : " << idx.size() << std::endl;
  };

  void init_network()
  {
    // Manually loading torch cuda DLL because it is not loading on its own
    LoadLibraryA("torch_cuda.dll");

    // device = torch::kCPU;

    if (torch::cuda::is_available())
    {
      std::cout << "CUDA available! \nTraining on CUDA" << std::endl;
      device = torch::kCUDA;
    }
    else
    {
      std::cout << "CUDA not available" << std::endl;
    }

    // LOAD MODEL //

    network = std::make_shared<Net>();
    network->to(device);
  }
};

std::vector<torch::Tensor> process_data_in(NetworkData net)
{
  std::vector<torch::Tensor> in_tensor;
  for (int i = 0; i < net.idx.size(); i++)
  {
    std::vector<float> temp;
    // temp.push_back(net.m_eye.x);
    // temp.push_back(net.m_eye.y);
    // temp.push_back(net.m_eye.z);
    temp.push_back(net.m_ray_vec[net.idx[i]].x);
    temp.push_back(net.m_ray_vec[net.idx[i]].y);
    temp.push_back(net.m_hit_data[net.idx[i]]);
    temp.push_back(net.m_light_vec.x);
    temp.push_back(net.m_light_vec.y);
    temp.push_back(net.m_light_vec.z);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sample = torch::from_blob(&temp[0], {10}, options);
    in_tensor.push_back(sample.clone());
    temp.clear();
  }
  return in_tensor;
}

std::vector<torch::Tensor> process_data_out(NetworkData net)
{
  std::vector<torch::Tensor> out_tensor;
  for (int i = 0; i < net.idx.size(); i++)
  {
    std::vector<float> temp;
    temp.push_back(clamp(net.m_result_buffer[net.idx[i]].x, 0.0f, 1.0f));
    temp.push_back(clamp(net.m_result_buffer[net.idx[i]].y, 0.0f, 1.0f));
    temp.push_back(clamp(net.m_result_buffer[net.idx[i]].z, 0.0f, 1.0f));
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sample = torch::from_blob(&temp[0], {3}, options);
    out_tensor.push_back(sample.clone());
    temp.clear();
  }
  return out_tensor;
}

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
  // Declare 2 vectors of tensors for input and output
  std::vector<torch::Tensor> in_data, out_data;

public:
  // Constructor
  CustomDataset(NetworkData net)
  {
    in_data = process_data_in(net);
    out_data = process_data_out(net);
  };

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override
  {
    torch::Tensor sample_in = in_data.at(index);
    torch::Tensor sample_out = out_data.at(index);
    return {sample_in.clone(), sample_out.clone()};
  };

  // Return the length of data
  torch::optional<size_t> size() const override
  {
    return out_data.size();
  };
};

void network(NetworkData data, int batchsize = 16)
{
  int epochs = 10;
  float lr = 0.01;
  data.training = true;

  // Manually loading torch cuda DLL because it is not loading on its own
  // LoadLibraryA("torch_cuda.dll");

  torch::optim::SGD optimizer(data.network->parameters(), lr);
  auto dataset = CustomDataset(data);
  auto data_loader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batchsize).workers(1));

  // Input Values
  //  torch::Tensor input_tensors;
  //  torch::Tensor output_tensors;
  for (size_t epoch = 1; epoch <= epochs; ++epoch)
  {
    size_t batch_index = 0;
    for (auto &batch : *data_loader)
    {
      // std::cout << typeid(batch[0].data).name() << " " << batch.size() << std::endl;

      std::vector<torch::Tensor> x_data, y_data;
      torch::Tensor batch_data, batch_target;

      for (int i = 0; i < batch.size(); i++)
      {
        x_data.push_back(batch[0].data);
        y_data.push_back(batch[0].target);
      }

      batch_data = torch::stack(x_data);
      batch_target = torch::stack(y_data);
      batch_data = batch_data.to(data.device);
      batch_target = batch_target.to(data.device);

      // std::cout << batch_data.device() << "    " << batch_target.device()  << std::endl ;

      optimizer.zero_grad();

      torch::Tensor preds = data.network->forward(batch_data);
      torch::Tensor loss = torch::mse_loss(preds, batch_target);

      loss.backward();
      optimizer.step();

      if (++batch_index % 1 == 0)
      {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
      }
    }
  }

  torch::save(data.network, "net.pt");
  std::cout << "Network saved" << std::endl;
  data.training = false;
};

float3 *NetworkData::m_x;
float2 *NetworkData::m_ray_vec;
float *NetworkData::m_hit_data;
float3 NetworkData::m_light_vec;
float3 *NetworkData::m_result_buffer;
int NetworkData::batchsize;
int NetworkData::buffersize;

std::vector<int> NetworkData::idx;

void append_data(std::ofstream &file, NetworkData data, float theta_i, float phi_i)
{
  for (int i = 0; i < data.idx.size(); i++)
  {
    file << data.m_x[data.idx[i]].x << ","
         << data.m_x[data.idx[i]].y << ","
         << data.m_x[data.idx[i]].z << ","
         << theta_i << ","
         << phi_i << ","
         << data.m_ray_vec[data.idx[i]].x << ","
         << data.m_ray_vec[data.idx[i]].y << ","
         << clamp(data.m_result_buffer[data.idx[i]].x, 0.0f, 1.0f) << ","
         << clamp(data.m_result_buffer[data.idx[i]].y, 0.0f, 1.0f) << ","
         << clamp(data.m_result_buffer[data.idx[i]].z, 0.0f, 1.0f) << std::endl;
  }
}