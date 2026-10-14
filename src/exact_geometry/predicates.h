#pragma once
// MIT: repository LICENSE. Independent native implementation of the mapped
// orientation/contact semantics in experiments/wrap/exact_geometry_parity.json.
// Shewchuk's expansion arithmetic: https://www.cs.cmu.edu/~quake/robust.html
// His predicates.c is public domain; no service source is imported or built.
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define CUMESH_GEOMETRY_HD __host__ __device__ inline
#else
#define CUMESH_GEOMETRY_HD inline
#endif

namespace cumesh::exact_geometry {
struct Point {
  double x, y, z;
  CUMESH_GEOMETRY_HD double operator[](int axis) const {
    return axis == 0 ? x : (axis == 1 ? y : z);
  }
};
struct Two {
  double main, tail;
};
// Explicit CUDA RN instructions confine strict arithmetic to the predicates;
// QEM cost/solver translation units retain their existing arithmetic flags.
CUMESH_GEOMETRY_HD double add(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dadd_rn(a, b);
#else
  volatile double value = a + b;
  return value;
#endif
}
CUMESH_GEOMETRY_HD double sub(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dsub_rn(a, b);
#else
  volatile double value = a - b;
  return value;
#endif
}
CUMESH_GEOMETRY_HD double mul(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dmul_rn(a, b);
#else
  volatile double value = a * b;
  return value;
#endif
}
CUMESH_GEOMETRY_HD Point difference(Point a, Point b) {
  return {sub(a.x, b.x), sub(a.y, b.y), sub(a.z, b.z)};
}
CUMESH_GEOMETRY_HD Point cross(Point a, Point b) {
  return {sub(mul(a.y, b.z), mul(a.z, b.y)), sub(mul(a.z, b.x), mul(a.x, b.z)),
          sub(mul(a.x, b.y), mul(a.y, b.x))};
}
CUMESH_GEOMETRY_HD Two two_sum(double a, double b) {
  double x = add(a, b), bv = sub(x, a), av = sub(x, bv);
  return {x, add(sub(a, av), sub(b, bv))};
}
CUMESH_GEOMETRY_HD Two two_product(double a, double b) {
  constexpr double splitter = 134217729.0;
  double x = mul(a, b), c = mul(splitter, a), abig = sub(c, a),
         ahi = sub(c, abig), alo = sub(a, ahi);
  c = mul(splitter, b);
  double bbig = sub(c, b), bhi = sub(c, bbig), blo = sub(b, bhi);
  double err1 = sub(x, mul(ahi, bhi)), err2 = sub(err1, mul(alo, bhi)),
         err3 = sub(err2, mul(ahi, blo));
  return {x, sub(mul(alo, blo), err3)};
}
CUMESH_GEOMETRY_HD Point row(Point a, Point b, Point c, Point d, int index) {
  return index == 0 ? a : (index == 1 ? b : (index == 2 ? c : d));
}
CUMESH_GEOMETRY_HD int orient3d_exact(Point a, Point b, Point c, Point d) {
  constexpr int capacity = 96;
  double expansion[capacity];
  int length = 0;
  for (int term = 0; term < 6; ++term) {
    int i = 0, j = 1, k = 2, l = 3;
    double sign = 1.0;
    if (term == 1) {
      j = 2;
      k = 1;
      sign = -1.0;
    } else if (term == 2) {
      j = 3;
      k = 1;
      l = 2;
    } else if (term == 3) {
      i = 1;
      j = 2;
      k = 0;
      l = 3;
    } else if (term == 4) {
      i = 1;
      j = 3;
      k = 0;
      l = 2;
      sign = -1.0;
    } else if (term == 5) {
      i = 2;
      j = 3;
      k = 0;
      l = 1;
    }
    Point ri = row(a, b, c, d, i), rj = row(a, b, c, d, j);
    double zk = mul(sign, row(a, b, c, d, k).z),
           zl = mul(-sign, row(a, b, c, d, l).z);
    Two forward = two_product(ri.x, rj.y), backward = two_product(rj.x, ri.y);
    for (int item = 0; item < 4; ++item) {
      double u =
          item == 0
              ? forward.tail
              : (item == 1 ? forward.main
                           : (item == 2 ? -backward.tail : -backward.main));
      for (int factor = 0; factor < 2; ++factor) {
        Two product = two_product(u, factor == 0 ? zk : zl);
        for (int part = 0; part < 2; ++part) {
          // Match the source's product[0] (main), then product[1] (tail).
          double carry = part == 0 ? product.main : product.tail;
          int index = 0;
          for (int position = 0; position < length; ++position) {
            Two step = two_sum(carry, expansion[position]);
            carry = step.main;
            if (step.tail != 0.0) {
              if (index == capacity)
                return 2;
              expansion[index++] = step.tail;
            }
          }
          if (carry != 0.0) {
            if (index == capacity)
              return 2;
            expansion[index++] = carry;
          }
          length = index;
        }
      }
    }
  }
  return length == 0 ? 0 : (expansion[length - 1] > 0.0 ? 1 : -1);
}
CUMESH_GEOMETRY_HD int orient3d(Point a, Point b, Point c, Point d) {
  Point ad = difference(a, d), bd = difference(b, d), cd = difference(c, d);
  double bc = mul(bd.x, cd.y), cb = mul(cd.x, bd.y), ca = mul(cd.x, ad.y),
         ac = mul(ad.x, cd.y), ab = mul(ad.x, bd.y), ba = mul(bd.x, ad.y);
  double determinant = add(add(mul(ad.z, sub(bc, cb)), mul(bd.z, sub(ca, ac))),
                           mul(cd.z, sub(ab, ba)));
  double permanent =
      add(add(mul(add(std::fabs(bc), std::fabs(cb)), std::fabs(ad.z)),
              mul(add(std::fabs(ca), std::fabs(ac)), std::fabs(bd.z))),
          mul(add(std::fabs(ab), std::fabs(ba)), std::fabs(cd.z)));
  constexpr double unit = 0x1p-53;
  double bound = mul(mul(add(7.0, mul(56.0, unit)), unit), permanent);
  if (determinant > bound)
    return 1;
  if (-determinant > bound)
    return -1;
  if (permanent == 0.0)
    return 0;
  return orient3d_exact(a, b, c, d);
}
CUMESH_GEOMETRY_HD int orient2d(int axis, Point a, Point b, Point c) {
  int first = axis == 0 ? 1 : (axis == 1 ? 2 : 0),
      second = axis == 0 ? 2 : (axis == 1 ? 0 : 1);
  double left = mul(sub(a[first], c[first]), sub(b[second], c[second])),
         right = mul(sub(a[second], c[second]), sub(b[first], c[first]));
  double determinant = sub(left, right),
         sum = add(std::fabs(left), std::fabs(right));
  constexpr double unit = 0x1p-53;
  double bound = mul(mul(add(3.0, mul(16.0, unit)), unit), sum);
  if (determinant > bound)
    return 1;
  if (-determinant > bound)
    return -1;
  if (sum == 0.0)
    return 0;
  int exact =
      orient3d_exact({a[first], a[second], 0.0}, {b[first], b[second], 0.0},
                     {c[first], c[second], 0.0}, {0.0, 0.0, 1.0});
  return exact == 2 ? 2 : -exact;
}
CUMESH_GEOMETRY_HD int dominant_axis(Point n) {
  double x = std::fabs(n.x), y = std::fabs(n.y), z = std::fabs(n.z);
  return x >= y && x >= z ? 0 : (y >= z ? 1 : 2);
}
CUMESH_GEOMETRY_HD bool near_underflow(Point a, Point b, Point c) {
  constexpr double guard = 0x1p-250;
  for (int axis = 0; axis < 3; ++axis)
    if ((a[axis] != 0.0 && std::fabs(a[axis]) < guard) ||
        (b[axis] != 0.0 && std::fabs(b[axis]) < guard) ||
        (c[axis] != 0.0 && std::fabs(c[axis]) < guard))
      return true;
  return false;
}
CUMESH_GEOMETRY_HD bool zero_area(Point a, Point b, Point c) {
  Point n = cross(difference(b, a), difference(c, a));
  return n.x == 0.0 && n.y == 0.0 && n.z == 0.0;
}
CUMESH_GEOMETRY_HD int segment_projected(int axis, Point p, Point q, Point u,
                                         Point v, Point w) {
  int side = orient2d(axis, u, v, w);
  if (side == 0 || side == 2)
    return 2;
  bool unsure = false;
  Point tri[3] = {u, v, w};
  for (int edge = 0; edge < 3; ++edge) {
    int e = orient2d(axis, tri[edge], tri[(edge + 1) % 3], p),
        f = orient2d(axis, tri[edge], tri[(edge + 1) % 3], q);
    if (e == -side && f == -side)
      return 0;
    if (e == 2 || f == 2)
      unsure = true;
  }
  int l1 = orient2d(axis, p, q, u), l2 = orient2d(axis, p, q, v),
      l3 = orient2d(axis, p, q, w);
  if ((l1 == 1 && l2 == 1 && l3 == 1) || (l1 == -1 && l2 == -1 && l3 == -1))
    return 0;
  if (l1 == 2 || l2 == 2 || l3 == 2)
    unsure = true;
  return unsure ? 2 : 1;
}
CUMESH_GEOMETRY_HD int segment_in_plane(Point p, Point q, Point u, Point v,
                                        Point w) {
  return segment_projected(
      dominant_axis(cross(difference(v, u), difference(w, u))), p, q, u, v, w);
}
CUMESH_GEOMETRY_HD bool projections_disjoint(int axis, Point a, Point b,
                                             Point c, Point u, Point v,
                                             Point w) {
  return segment_projected(axis, a, b, u, v, w) == 0 &&
         segment_projected(axis, b, c, u, v, w) == 0 &&
         segment_projected(axis, c, a, u, v, w) == 0 &&
         segment_projected(axis, u, v, a, b, c) == 0 &&
         segment_projected(axis, v, w, a, b, c) == 0 &&
         segment_projected(axis, w, u, a, b, c) == 0;
}
CUMESH_GEOMETRY_HD bool projections_shared_vertex(int axis, Point e0, Point e1,
                                                  Point f0, Point f1, Point a,
                                                  Point b, Point c, Point u,
                                                  Point v, Point w) {
  return segment_projected(axis, e0, e1, u, v, w) == 0 &&
         segment_projected(axis, f0, f1, a, b, c) == 0;
}
CUMESH_GEOMETRY_HD int segment_signed(Point p, Point q, Point u, Point v,
                                      Point w, int sp, int sq) {
  if (sp == 2 || sq == 2)
    return 2;
  if ((sp == 1 && sq == 1) || (sp == -1 && sq == -1))
    return 0;
  if (sp == 0 && sq == 0)
    return segment_in_plane(p, q, u, v, w);
  int t1 = orient3d(p, q, u, v), t2 = orient3d(p, q, v, w),
      t3 = orient3d(p, q, w, u);
  if ((t1 == 1 || t2 == 1 || t3 == 1) && (t1 == -1 || t2 == -1 || t3 == -1))
    return 0;
  if (t1 == 2 || t2 == 2 || t3 == 2)
    return 2;
  return 1;
}
CUMESH_GEOMETRY_HD int segment(Point p, Point q, Point u, Point v, Point w) {
  return segment_signed(p, q, u, v, w, orient3d(u, v, w, p),
                        orient3d(u, v, w, q));
}
CUMESH_GEOMETRY_HD int either(int first, int second) {
  return first == 1 || second == 1 ? 1 : (first == 2 || second == 2 ? 2 : 0);
}
CUMESH_GEOMETRY_HD int shared_edge(Point p, Point q, Point r1, Point r2) {
  int axis = dominant_axis(cross(difference(q, p), difference(r1, p))),
      o1 = orient2d(axis, p, q, r1), o2 = orient2d(axis, p, q, r2);
  if (o1 == 0 || o1 == 2)
    return 2;
  if (o2 == -o1)
    return 0;
  int plane = orient3d(p, q, r1, r2);
  if (plane == 2)
    return 2;
  if (plane != 0)
    return 0;
  if (o2 == o1)
    return 1;
  return 2;
}
CUMESH_GEOMETRY_HD int nonadjacent(Point a, Point b, Point c, Point u, Point v,
                                   Point w) {
  int sa = orient3d(u, v, w, a), sb = orient3d(u, v, w, b),
      sc = orient3d(u, v, w, c);
  if ((sa == 1 && sb == 1 && sc == 1) || (sa == -1 && sb == -1 && sc == -1))
    return 0;
  int su = orient3d(a, b, c, u), sv = orient3d(a, b, c, v),
      sw = orient3d(a, b, c, w);
  if ((su == 1 && sv == 1 && sw == 1) || (su == -1 && sv == -1 && sw == -1))
    return 0;
  int coplanar = sa == 0 && sb == 0 && sc == 0 ? 4 : 0;
  int result = either(segment_signed(a, b, u, v, w, sa, sb),
                      segment_signed(b, c, u, v, w, sb, sc));
  if (result != 1)
    result = either(result, either(segment_signed(c, a, u, v, w, sc, sa),
                                   segment_signed(u, v, a, b, c, su, sv)));
  if (result != 1)
    result = either(result, either(segment_signed(v, w, a, b, c, sv, sw),
                                   segment_signed(w, u, a, b, c, sw, su)));
  if (result == 2) {
    int first = dominant_axis(cross(difference(b, a), difference(c, a))),
        second = dominant_axis(cross(difference(v, u), difference(w, u)));
    if (projections_disjoint(first, a, b, c, u, v, w) ||
        (second != first && projections_disjoint(second, a, b, c, u, v, w)))
      result = 0;
  }
  return result + coplanar;
}
CUMESH_GEOMETRY_HD int pair(Point a, Point b, Point c, Point u, Point v,
                            Point w, int ga, int gb, int gc, int gu, int gv,
                            int gw) {
  bool sa = ga == gu || ga == gv || ga == gw,
       sb = gb == gu || gb == gv || gb == gw,
       sc = gc == gu || gc == gv || gc == gw;
  bool su = gu == ga || gu == gb || gu == gc,
       sv = gv == ga || gv == gb || gv == gc,
       sw = gw == ga || gw == gb || gw == gc;
  int shared = int(sa) + int(sb) + int(sc), other = int(su) + int(sv) + int(sw);
  if (zero_area(a, b, c) || zero_area(u, v, w) || near_underflow(a, b, c) ||
      near_underflow(u, v, w) || shared != other)
    return 2;
  if (shared == 0)
    return nonadjacent(a, b, c, u, v, w);
  if (shared == 3)
    return 1;
  if (shared == 2) {
    Point r1 = a, p = b, q = c;
    if (!sb) {
      r1 = b;
      p = c;
      q = a;
    } else if (!sc) {
      r1 = c;
      p = a;
      q = b;
    }
    Point r2 = u;
    if (!sv)
      r2 = v;
    else if (!sw)
      r2 = w;
    return shared_edge(p, q, r1, r2);
  }
  Point e0 = b, e1 = c;
  if (sb) {
    e0 = c;
    e1 = a;
  } else if (sc) {
    e0 = a;
    e1 = b;
  }
  Point f0 = v, f1 = w;
  if (sv) {
    f0 = w;
    f1 = u;
  } else if (sw) {
    f0 = u;
    f1 = v;
  }
  int result = either(segment(e0, e1, u, v, w), segment(f0, f1, a, b, c));
  if (result == 2) {
    int first = dominant_axis(cross(difference(b, a), difference(c, a))),
        second = dominant_axis(cross(difference(v, u), difference(w, u)));
    if (projections_shared_vertex(first, e0, e1, f0, f1, a, b, c, u, v, w) ||
        (second != first &&
         projections_shared_vertex(second, e0, e1, f0, f1, a, b, c, u, v, w)))
      result = 0;
  }
  return result;
}
} // namespace cumesh::exact_geometry
#undef CUMESH_GEOMETRY_HD
