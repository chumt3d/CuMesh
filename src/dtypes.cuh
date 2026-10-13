#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


namespace cumesh {


/**
 * A 3D vector class with overloaded operators and methods.
 */
struct __align__(16) Vec3f {
    float x, y, z;

    __device__ __forceinline__ Vec3f();
    __device__ __forceinline__ Vec3f(float x, float y, float z);
    __device__ __forceinline__ Vec3f(float3 v);
    __device__ __forceinline__ Vec3f operator+(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f& operator+=(const Vec3f& o);
    __device__ __forceinline__ Vec3f operator-(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f& operator-=(const Vec3f& o);
    __device__ __forceinline__ Vec3f operator*(float s) const;
    __device__ __forceinline__ Vec3f& operator*=(float s);
    __device__ __forceinline__ Vec3f operator/(float s) const;
    __device__ __forceinline__ Vec3f& operator/=(float s);
    __device__ __forceinline__ float dot(const Vec3f& o) const;
    __device__ __forceinline__ float norm() const;
    __device__ __forceinline__ float norm2() const;
    __device__ __forceinline__ Vec3f normalized() const;
    __device__ __forceinline__ void normalize();
    __device__ __forceinline__ Vec3f cross(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f slerp(const Vec3f& o, float t) const;
};


/**
 * QEM (Quadric Error Metric) class for mesh simplification.
 */
struct __align__(16) QEM
{
    // store upper triangle of symmetric 4x4 matrix:
    // e = [ 00, 01, 02, 03, 11, 12, 13, 22, 23, 33 ]
    double e[10];
    double weight;
    double padding; // initialize the 12-double, 16-byte-aligned Tensor row

    __device__ __forceinline__ QEM();
    __device__ __forceinline__ QEM operator+(const QEM& o) const;
    __device__ __forceinline__ QEM& operator+=(const QEM& o);
    __device__ __forceinline__ QEM operator-(const QEM& o) const;
    __device__ __forceinline__ QEM& operator-=(const QEM& o);
    __device__ __forceinline__ void zero();
    __device__ __forceinline__ void add_plane(double4 p, double plane_weight);
    __device__ __forceinline__ double evaluate(const Vec3f& p) const;
    __device__ __forceinline__ bool solve_optimal(float3 &out, double &err) const;
    __device__ __forceinline__ float3 solve_segment(float3 a, float3 b) const;
    __device__ __forceinline__ double normalized_error(const Vec3f& p) const;
};


__device__ __forceinline__ Vec3f::Vec3f() {
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}

__device__ __forceinline__ Vec3f::Vec3f(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

__device__ __forceinline__ Vec3f::Vec3f(float3 v) {
    x = v.x;
    y = v.y;
    z = v.z;
}


__device__ __forceinline__ Vec3f Vec3f::operator+(const Vec3f& o) const {
    return Vec3f(x + o.x, y + o.y, z + o.z);
}


__device__ __forceinline__ Vec3f& Vec3f::operator+=(const Vec3f& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator-(const Vec3f& o) const {
    return Vec3f(x - o.x, y - o.y, z - o.z);
}


__device__ __forceinline__ Vec3f& Vec3f::operator-=(const Vec3f& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator*(float s) const {
    return Vec3f(x * s, y * s, z * s);
}


__device__ __forceinline__ Vec3f& Vec3f::operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator/(float s) const {
    return Vec3f(x / s, y / s, z / s);
}


__device__ __forceinline__ Vec3f& Vec3f::operator/=(float s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
}


__device__ __forceinline__ float Vec3f::dot(const Vec3f& o) const {
    return x * o.x + y * o.y + z * o.z;
}


__device__ __forceinline__ float Vec3f::norm() const {
    return sqrtf(x * x + y * y + z * z);
}


__device__ __forceinline__ float Vec3f::norm2() const {
    return x * x + y * y + z * z;
}


__device__ __forceinline__ Vec3f Vec3f::normalized() const {
    float inv_norm = rsqrtf(x * x + y * y + z * z);
    return Vec3f(x * inv_norm, y * inv_norm, z * inv_norm);
}


__device__ __forceinline__ void Vec3f::normalize() {
    float inv_norm = rsqrtf(x * x + y * y + z * z);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
}


__device__ __forceinline__ Vec3f Vec3f::cross(const Vec3f& o) const {
    return Vec3f(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
}


__device__ __forceinline__ Vec3f Vec3f::slerp(const Vec3f& o, float t) const {
    float dot_prod = this->dot(o);
    dot_prod = fmaxf(fminf(dot_prod, 1.0f), -1.0f); // Clamp to [-1, 1]
    float theta = acosf(dot_prod) * t;
    Vec3f relative_vec = (o - (*this) * dot_prod).normalized();
    return (*this) * cosf(theta) + relative_vec * sinf(theta);
}


__device__ __forceinline__ QEM::QEM() {
    zero();
}


__device__ __forceinline__ QEM QEM::operator+(const QEM& o) const {
    QEM res;
    #pragma unroll
    for (int i = 0; i < 10; ++i) res.e[i] = e[i] + o.e[i];
    res.weight = weight + o.weight;
    return res;
}


__device__ __forceinline__ QEM& QEM::operator+=(const QEM& o) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] += o.e[i];
    weight += o.weight;
    return *this;
}


__device__ __forceinline__ QEM QEM::operator-(const QEM& o) const {
    QEM res;
    #pragma unroll
    for (int i = 0; i < 10; ++i) res.e[i] = e[i] - o.e[i];
    res.weight = weight - o.weight;
    return res;
}


__device__ __forceinline__ QEM& QEM::operator-=(const QEM& o) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] -= o.e[i];
    weight -= o.weight;
    return *this;
}

__device__ __forceinline__ void QEM::zero() {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] = 0.0;
    weight = padding = 0.0;
}


// Add plane p = (a,b,c,d) as outer product p * p^T
__device__ __forceinline__ void QEM::add_plane(double4 p, double plane_weight) {
    // upper triangle indices mapping:
    // (0,0)->e[0]
    // (0,1)->e[1]
    // (0,2)->e[2]
    // (0,3)->e[3]
    // (1,1)->e[4]
    // (1,2)->e[5]
    // (1,3)->e[6]
    // (2,2)->e[7]
    // (2,3)->e[8]
    // (3,3)->e[9]
    double a = p.x, b = p.y, c = p.z, d = p.w;
    weight += plane_weight;
    e[0] += plane_weight * a * a;
    e[1] += plane_weight * a * b;
    e[2] += plane_weight * a * c;
    e[3] += plane_weight * a * d;
    e[4] += plane_weight * b * b;
    e[5] += plane_weight * b * c;
    e[6] += plane_weight * b * d;
    e[7] += plane_weight * c * c;
    e[8] += plane_weight * c * d;
    e[9] += plane_weight * d * d;
}


// Evaluate v^T * Q * v for v = (x,y,z,1)
__device__ __forceinline__ double QEM::evaluate(const Vec3f& p) const {
    // compute v = [x,y,z,1]
    double x = p.x, y = p.y, z = p.z, w = 1.0;
    // expand symmetric multiplication using stored upper triangular
    // result = sum_{i<=j} M_ij * v_i * v_j * (1 if i==j else 2)
    double res = 0.0;
    // (0,0)
    res += e[0] * x * x;
    // (0,1) and (1,0)
    res += 2.0f * e[1] * x * y;
    // (0,2)
    res += 2.0f * e[2] * x * z;
    // (0,3)
    res += 2.0f * e[3] * x * w;
    // (1,1)
    res += e[4] * y * y;
    // (1,2)
    res += 2.0f * e[5] * y * z;
    // (1,3)
    res += 2.0f * e[6] * y * w;
    // (2,2)
    res += e[7] * z * z;
    // (2,3)
    res += 2.0f * e[8] * z * w;
    // (3,3)
    res += e[9] * w * w;
    return res;
}


// Weighted error and relative LDL pivots follow meshoptimizer b233043,
// src/simplifier.cpp quadricError/quadricSolve (MIT), without copying its CPU
// implementation. Every proposal rounds to actual fp32 before its predicates.
__device__ __forceinline__ double QEM::normalized_error(const Vec3f& p) const {
    return weight > 0.0 && isfinite(weight) ? fabs(evaluate(p)) / weight : INFINITY;
}

__device__ __forceinline__ bool QEM::solve_optimal(float3& out, double& err) const {
    const double relative_pivot_tolerance = 1e-6; // meshoptimizer b233043
    double eps=relative_pivot_tolerance*weight, d0=e[0];
    if (!(d0 > eps)) return false;
    double l10=e[1]/d0,l20=e[2]/d0,d1=e[4]-e[1]*l10;
    if (!(d1 > eps)) return false;
    double t=e[5]-e[2]*l10,l21=t/d1,d2=e[7]-e[2]*l20-t*l21;
    if (!(d2 > eps)) return false;
    double y0=-e[3],y1=-e[6]-l10*y0,y2=-e[8]-l20*y0-l21*y1;
    double z=y2/d2,y=y1/d1-l21*z,x=y0/d0-l10*y-l20*z;
    out=make_float3(float(x),float(y),float(z));
    err=normalized_error(Vec3f(out));
    return isfinite(out.x)&&isfinite(out.y)&&isfinite(out.z)&&isfinite(err);
}

__device__ __forceinline__ float3 QEM::solve_segment(float3 a,float3 b) const {
    double x=a.x,y=a.y,z=a.z,dx=double(b.x)-x,dy=double(b.y)-y,dz=double(b.z)-z;
    double ax=e[0]*dx+e[1]*dy+e[2]*dz,ay=e[1]*dx+e[4]*dy+e[5]*dz,az=e[2]*dx+e[5]*dy+e[7]*dz;
    double curvature=dx*ax+dy*ay+dz*az;
    double gradient=dx*(e[0]*x+e[1]*y+e[2]*z+e[3])+dy*(e[1]*x+e[4]*y+e[5]*z+e[6])+dz*(e[2]*x+e[5]*y+e[7]*z+e[8]);
    double t=curvature > 0.0 ? fmax(0.0,fmin(1.0,-gradient/curvature)) : 0.5;
    return make_float3(float(x+t*dx),float(y+t*dy),float(z+t*dz));
}


} // namespace cumesh
