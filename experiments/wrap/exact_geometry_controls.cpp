// Host controls for the same host/device contact predicates used by CUDA.
// Each case exercises an observable admission decision, without a service
// import.
#include "../../src/exact_geometry/predicates.h"
#include <array>
#include <iostream>
#include <stdexcept>
using cumesh::exact_geometry::Point;
struct Case {
  const char *name;
  std::array<Point, 6> points;
  std::array<int, 6> groups;
  int expected;
};
int main() {
  const Point a{0, 0, 0}, b{1, 0, 0}, c{0, 1, 0};
  const Case cases[] = {
      {"legal_shared_edge", {a, b, c, a, b, {0, -1, 0}}, {0, 1, 2, 0, 1, 3}, 0},
      {"coplanar_shared_edge_overlap",
       {a, b, c, a, b, {0.25, 0.25, 0}},
       {0, 1, 2, 0, 1, 3},
       1},
      {"legal_single_vertex",
       {a, b, c, a, {-1, 0, 1}, {0, -1, 1}},
       {0, 1, 2, 0, 3, 4},
       0},
      {"shared_vertex_overlap",
       {a, b, c, a, {0.5, 0, 0}, {0, 0.5, 0}},
       {0, 1, 2, 0, 3, 4},
       1},
      {"nonadjacent_crossing",
       {a, b, c, {0.25, 0.2, -1}, {0.25, 0.2, 1}, {0.25, 0.8, 0}},
       {0, 1, 2, 3, 4, 5},
       1},
      {"parallel_disjoint",
       {a, b, c, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}},
       {0, 1, 2, 3, 4, 5},
       0},
      {"coplanar_disjoint",
       {a, b, c, {3, 0, 0}, {4, 0, 0}, {3, 1, 0}},
       {0, 1, 2, 3, 4, 5},
       4},
      {"duplicate_triangle", {a, b, c, a, b, c}, {0, 1, 2, 0, 1, 2}, 1},
      {"zero_area_refused",
       {a, b, {2, 0, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}},
       {0, 1, 2, 3, 4, 5},
       2},
      {"underflow_refused",
       {a, b, c, {0, 0, 0x1p-300}, {1, 0, 1}, {0, 1, 1}},
       {0, 1, 2, 3, 4, 5},
       2},
      {"tiny_f32_parallel_disjoint",
       {a, b, c, {0, 0, 0x1p-140}, {1, 0, 0x1p-140}, {0, 1, 0x1p-140}},
       {0, 1, 2, 3, 4, 5},
       0}};
  for (const auto &test : cases) {
    auto p = test.points;
    auto g = test.groups;
    int actual = cumesh::exact_geometry::pair(
        p[0], p[1], p[2], p[3], p[4], p[5], g[0], g[1], g[2], g[3], g[4], g[5]);
    if (actual != test.expected)
      throw std::runtime_error(std::string(test.name) + ": got " +
                               std::to_string(actual) + ", expected " +
                               std::to_string(test.expected));
  }
  std::cout << "11 exact contact controls passed\n";
}
