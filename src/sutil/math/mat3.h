// Code written by Jeppe Revall Frisvad,
// Technical University of Denmark, 2017.

#ifndef MAT3_H
#define MAT3_H

#include <src/sutil/math/vec_math.h>

// CUDA-style constructors
struct mat3;
mat3 make_mat3(const float a);
mat3 make_mat3(const float3 &v1, const float3 &v2, const float3 &v3);

struct mat3
{
  // data
  float3 x, y, z;

  // subscript operator
  float &operator[](unsigned int i) { return *(&x.x + i); }
  const float &operator[](unsigned int i) const { return *(&x.x + i); }

  // assignment operators
  template <class T>
  const mat3 &operator*=(T k)
  {
    x *= k;
    y *= k;
    z *= k;
    return *this;
  }
  template <class T>
  const mat3 &operator/=(T k)
  {
    x /= k;
    y /= k;
    z /= k;
    return *this;
  }
  template <class T>
  const mat3 &operator+=(T k)
  {
    x += k;
    y += k;
    z += k;
    return *this;
  }
  template <class T>
  const mat3 &operator-=(T k)
  {
    x -= k;
    y -= k;
    z -= k;
    return *this;
  }
  const mat3 &operator*=(const mat3 &m)
  {
    x *= m.x;
    y *= m.y;
    z *= m.z;
    return *this;
  }
  const mat3 &operator/=(const mat3 &m)
  {
    x = x / m.x;
    y = y / m.y;
    z = z / m.z;
    return *this;
  }
  const mat3 &operator+=(const mat3 &m)
  {
    x += m.x;
    y += m.y;
    z += m.z;
    return *this;
  }
  const mat3 &operator-=(const mat3 &m)
  {
    x -= m.x;
    y -= m.y;
    z -= m.z;
    return *this;
  }

  // negation
  const mat3 operator-() const { return make_mat3(-x, -y, -z); }

  // binary operators on vectors
  const mat3 operator*(const mat3 &m) const
  {
    return make_mat3(x * m.x, y * m.y, z * m.z);
  }
  const mat3 operator/(const mat3 &m) const
  {
    return make_mat3(x / m.x, y / m.y, z / m.z);
  }
  const mat3 operator+(const mat3 &m) const
  {
    return make_mat3(x + m.x, y + m.y, z + m.z);
  }
  const mat3 operator-(const mat3 &m) const
  {
    return make_mat3(x - m.x, y - m.y, z - m.z);
  }

  // binary operators on vector and scalar
  template <class T>
  const mat3 operator*(T k) const
  {
    return make_mat3(x * static_cast<float>(k), y * static_cast<float>(k),
                     z * static_cast<float>(k));
  }
  template <class T>
  const mat3 operator/(T k) const
  {
    return make_mat3(x / static_cast<float>(k), y / static_cast<float>(k),
                     z / static_cast<float>(k));
  }
};

// CUDA-style constructors
inline mat3 make_mat3(const float a)
{
  mat3 m;
  m.x = make_float3(a, 0.0f, 0.0f);
  m.y = make_float3(0.0f, a, 0.0f);
  m.z = make_float3(0.0f, 0.0f, a);
  return m;
}
inline mat3 make_mat3(const float3 &d)
{
  mat3 m;
  m.x = make_float3(d.x, 0.0f, 0.0f);
  m.y = make_float3(0.0f, d.y, 0.0f);
  m.z = make_float3(0.0f, 0.0f, d.z);
  return m;
}
inline mat3 make_mat3(const float3 &v1, const float3 &v2, const float3 &v3)
{
  mat3 m;
  m.x = v1;
  m.y = v2;
  m.z = v3;
  return m;
}

// multiplication of a scalar from the right side
inline const mat3 operator*(double k, const mat3 &m) { return m * k; }
inline const mat3 operator*(float k, const mat3 &m) { return m * k; }
inline const mat3 operator*(int k, const mat3 &m) { return m * k; }

// other commonly used operations
inline const mat3 transpose(const mat3 &m)
{
  return make_mat3(make_float3(m.x.x, m.y.x, m.z.x),
                   make_float3(m.x.y, m.y.y, m.z.y),
                   make_float3(m.x.z, m.y.z, m.z.z));
}

inline const mat3 mult(const mat3 &a, const mat3 &b)
{
  const mat3 b_t = transpose(b);
  return make_mat3(
      make_float3(dot(a.x, b_t.x), dot(a.x, b_t.y), dot(a.x, b_t.z)),
      make_float3(dot(a.y, b_t.x), dot(a.y, b_t.y), dot(a.y, b_t.z)),
      make_float3(dot(a.z, b_t.x), dot(a.z, b_t.y), dot(a.z, b_t.z)));
}

SUTILFN const float3 mult(const mat3 &m, const float3 &v)
{
  return make_float3(dot(m.x, v), dot(m.y, v), dot(m.z, v));
}

inline const float3 diag(const mat3 &m)
{
  return make_float3(m.x.x, m.y.y, m.z.z);
}

inline float sum(const mat3 &m)
{
  float result = 0.0f;
  for (unsigned int i = 0; i < 9; ++i)
    result += m[i];
  return result;
}

inline mat3 identity_mat3() { return make_mat3(1.0f); }

inline bool invert(const mat3 &m, mat3 &m_inv)
{
  // Based on Cramer's rule (Akenine-M�ller et al. 2008, Section A.3.1) and
  // Cedrick Collomb. A tutorial on inverting 3 by 3 matrices with cross
  // products. http://www.emptyloop.com/technotes/
  mat3 a = transpose(m);
  float3 b = cross(a.y, a.z);
  float det = dot(a.x, b);

  if (det < 1.0e-8f)
    return false;

  m_inv = make_mat3(b, cross(a.z, a.x), cross(a.x, a.y)) * (1.0f / det);
  return true;
}

#endif // MAT3_H
