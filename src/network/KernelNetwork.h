#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <optix.h>
#include <cuda/GeometryData.h>

#include <src/sutil/math/Matrix.h>
#include "src/device/cuda/color.h"

template <unsigned int input_n = 6, unsigned int n_embedding = infnetwork::embedding_size>
__device__ __inline__ void embed(float input[input_n],
                                 float output[2 * n_embedding * input_n])
{
  float freq_bands8[] = {1., 19.142857, 37.285713, 55.42857,
                         73.57143, 91.71428, 109.85714, 128.};
  float freq_bands4[] = {1., 3.3333333, 5.6666665, 8.};

  for (unsigned int i = 0; i < n_embedding; i++)
  {
    for (unsigned int j = 0; j < 2; j++)
    {
      for (unsigned int k = 0; k < input_n; k++)
      {
        float input_val = input[k];
        float embedding = j % 2 == 0 ? sinf(input_val * freq_bands4[i])
                                     : cosf(input_val * freq_bands4[i]);
        unsigned int index = i * 2 * input_n + j * input_n + k;
        output[index] = embedding;
      }
    }
  }
}

__device__ __inline__ sutil::Matrix3x3 skew(const float3 &vec)
{
  sutil::Matrix3x3 res;
  res[0] = 0;
  res[1] = -vec.z;
  res[2] = vec.y;

  res[3] = vec.z;
  res[4] = 0;
  res[5] = -vec.x;

  res[6] = -vec.y;
  res[7] = vec.x;
  res[8] = 0;
  return res;
}

__device__ __inline__ sutil::Matrix3x3 getRotation(const float3 &from,
                                                   const float3 &to)
{

  const float3 v = cross(from, to);
  const float c = dot(from, to);

  sutil::Matrix3x3 res = sutil::Matrix3x3::identity();
  sutil::Matrix3x3 skewV = skew(v);
  return res + skewV + (skewV * skewV) * (1.f / (1.f + c));
}

__device__ __inline__ float beckmann(float s, float cos_theta_h)
{
  float alpha = acosf(cos_theta_h);
  float cos2 = cosf(alpha) * cosf(alpha);
  float cos4 = cos2 * cos2;
  float num = expf(-(1 - cos2) / (cos2 * s * s));
  float denom = M_PIf * s * s * cos4;
  return num / denom;
}

template <unsigned int n>
__device__ __inline__ void compute_beckmann(const BufferView<float> &s,
                                            float output[n], float cos_theta_h)
{
  output[n - 1] = beckmann(s[0], cos_theta_h);
}

template <unsigned int n>
__device__ __inline__ void compute_phong(const BufferView<float> &shininess,
                                         const BufferView<float> &scale,
                                         float output[n], float cos_theta_h)
{
  for (unsigned int i = 0; i < n; i++)
  {
    output[i] = powf(cos_theta_h, shininess[i]) * scale[i];
  }
}

__device__ __inline__ float swish(float x)
{
  return x * (1.f / (1.f + expf(-x)));
}

__device__ __inline__ float l_relu(float r, float x)
{
  return fmaxf(r * x, x);
}

template <unsigned int n>
__device__ __inline__ void swish(float input[n])
{
  for (unsigned int i = 0; i < n; i++)
  {
    input[i] = swish(input[i]);
  }
}

template <unsigned int n>
__device__ __inline__ void l_relu(float r, float input[n])
{
  for (unsigned int i = 0; i < n; i++)
  {
    input[i] = l_relu(r, input[i]);
  }
}

__device__ __inline__ double sigmoid(float n)
{
  return 1 / (1 + expf(-n));
}

template <unsigned int n>
__device__ __inline__ void sigmoid(float input[n])
{
  for (unsigned int i = 0; i < n; i++)
  {
    input[i] = sigmoid(input[i]);
  }
}

template <unsigned int n, unsigned int m>
__device__ __inline__ void dense(const BufferView<float> &w, const BufferView<float> &b,
                                 const float input[n], float output[m])
{
  for (unsigned int i = 0; i < m; ++i)
  {
    output[i] = 0.0;
    const unsigned int offset = i * n;
    for (unsigned int j = 0; j < n; ++j)
    {
      output[i] += w[offset + j] * input[j];
    }
    output[i] += b[i];
  }
}

template <unsigned int input_n = infnetwork::input_size, unsigned int output_n = infnetwork::output_size,
          unsigned int intermediate_n = infnetwork::layer_size>
__device__ __inline__ void evalNetwork(const struct infnetwork::data &network,
                                       float input[input_n],
                                       float output[output_n])
{
  float intermediateRes0[intermediate_n] = {0.f};
  float intermediateRes1[intermediate_n] = {0.f};
  dense<input_n, intermediate_n>(network.w0, network.b0, input, intermediateRes0);
  l_relu<output_n>(0.01f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w1, network.b1, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.01f, intermediateRes1);

  dense<intermediate_n, intermediate_n>(network.w2, network.b2, intermediateRes1, intermediateRes0);
  l_relu<output_n>(0.01f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w3, network.b3, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.0f, intermediateRes1);

  dense<intermediate_n, intermediate_n>(network.w4, network.b4, intermediateRes1, intermediateRes0);
  l_relu<output_n>(0.0f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w5, network.b5, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.0f, intermediateRes1);

  dense<intermediate_n, intermediate_n>(network.w6, network.b6, intermediateRes1, intermediateRes0);
  l_relu<output_n>(0.0f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w7, network.b7, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.0f, intermediateRes1);

  dense<intermediate_n, output_n>(network.w8, network.b8, intermediateRes1, output);
  sigmoid<output_n>(output);

  // for (unsigned int i = 0 ; i < output_n; i++) {
  //   output[i] = expf(output[i]) - 1;
  //   output[i] /= 1e6; // network.flux[i];
  // }
};

template <unsigned int input_n = infnetwork::input_size, unsigned int output_n = infnetwork::output_size,
          unsigned int intermediate_n = infnetwork::layer_size>
__device__ __inline__ void evalNetwork(const struct infnetwork::data &network,
                                       BufferView<float> input,
                                       BufferView<float> output)
{
  float intermediateRes0[intermediate_n] = {0.f};
  float intermediateRes1[intermediate_n] = {0.f};
  dense<input_n, intermediate_n>(network.w0, network.b0, input, intermediateRes0);
  l_relu<output_n>(0.01f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w1, network.b1, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.01f, intermediateRes1);

  dense<intermediate_n, intermediate_n>(network.w2, network.b2, intermediateRes1, intermediateRes0);
  l_relu<output_n>(0.01f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w3, network.b3, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.0f, intermediateRes1);

  dense<intermediate_n, intermediate_n>(network.w4, network.b4, intermediateRes1, intermediateRes0);
  l_relu<output_n>(0.0f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w5, network.b5, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.0f, intermediateRes1);

  dense<intermediate_n, intermediate_n>(network.w6, network.b6, intermediateRes1, intermediateRes0);
  l_relu<output_n>(0.0f, intermediateRes0);

  dense<intermediate_n, intermediate_n>(network.w7, network.b7, intermediateRes0, intermediateRes1);
  l_relu<output_n>(0.0f, intermediateRes1);

  dense<intermediate_n, output_n>(network.w8, network.b8, intermediateRes1, output);
  sigmoid<output_n>(output);

  // for (unsigned int i = 0 ; i < output_n; i++) {
  //   output[i] = expf(output[i]) - 1;
  //   output[i] /= 1e6; // network.flux[i];
  // }
};

__device__ __inline__ float3 evalNetwork(const float3 &xi, const float3 &wi, const float3 &ni,
                                         const float3 &xo, const float3 &wo, const float3 &no,
                                         const struct infnetwork::data &network)
{
  sutil::Matrix3x3 Ri = getRotation(ni, network.normal);
  sutil::Matrix3x3 Ro = getRotation(no, network.normal);
  float input[infnetwork::input_size];
  // Input xi
  input[0] = 0.0;
  input[1] = 0.0;
  // Input xo
  input[2] = (xo.x - xi.x) * 1.f;
  input[3] = (xo.z - xi.z) * 1.f;
  // Input wi
  float input_dir[6];
  float3 wi_network = normalize(Ri * wi);
  input_dir[0] = wi_network.x;
  input_dir[1] = wi_network.y;
  input_dir[2] = wi_network.z;
  // Input wo
  float3 wo_network = normalize(Ro * wo);
  input_dir[3] = wo_network.x;
  input_dir[4] = wo_network.y;
  input_dir[5] = wo_network.z;

  // if (dot(ni, make_float3(0,1,0)) > 0.0f) {
  //   // Embed directional input
  //   embed(input_dir, &input[4]);
  //   // Compute half-vector
  //   float3 hr = normalize(wi_network + wo_network);
  //   // const float beckmann_term = beckmann(network.beckmann_shininess[0], hr.z);
  //   // input[infnetwork::input_size - 1] = beckmann_term;

  float output[infnetwork::output_size];
  evalNetwork(network, input, output);
  //   return color::spectrum_to_rgb_fixed(output);
  // } else {
  //   float dist = sqrt(input[2] * input[2] + input[3] * input[3]);
  //   if (dist < 0.2f) {
  //     return make_float3(1,1,1);
  //   }
  // }

  return make_float3(output[0], output[1], output[2]);
}

#endif

__device__ __inline__ void init_denoiser_weights()
{
}