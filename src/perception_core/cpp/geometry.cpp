// Exact rotated-rectangle intersection, with cached corners and AABB rejection.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
constexpr double eps = 1e-12;
struct Point { double x, y; };
struct Rectangle {
    std::array<Point, 4> corners;
    double xmin, xmax, ymin, ymax, area;
};

Rectangle rectangle(const double* box) {
    const double c = std::cos(box[6]), s = std::sin(box[6]);
    const double l = box[3] / 2, w = box[4] / 2;
    const std::array<Point, 4> local = {{{l,w}, {-l,w}, {-l,-w}, {l,-w}}};
    Rectangle r;
    for (int k = 0; k < 4; ++k) {
        r.corners[k] = {box[0] + c * local[k].x - s * local[k].y,
                        box[1] + s * local[k].x + c * local[k].y};
    }
    r.xmin = r.xmax = r.corners[0].x;
    r.ymin = r.ymax = r.corners[0].y;
    for (const auto& p : r.corners) {
        r.xmin = std::min(r.xmin, p.x); r.xmax = std::max(r.xmax, p.x);
        r.ymin = std::min(r.ymin, p.y); r.ymax = std::max(r.ymax, p.y);
    }
    r.area = box[3] * box[4];
    return r;
}
double cross(Point a, Point b, Point p) {
    return (b.x-a.x)*(p.y-a.y) - (b.y-a.y)*(p.x-a.x);
}
Point intersection(Point p, Point q, Point a, Point b) {
    const double denom = (p.x-q.x)*(a.y-b.y) - (p.y-q.y)*(a.x-b.x);
    if (std::abs(denom) < eps) return q;
    const double t = ((p.x-a.x)*(a.y-b.y) - (p.y-a.y)*(a.x-b.x)) / denom;
    return {p.x+t*(q.x-p.x), p.y+t*(q.y-p.y)};
}
double overlap(const Rectangle& a, const Rectangle& b) {
    if (a.xmax < b.xmin-eps || b.xmax < a.xmin-eps ||
        a.ymax < b.ymin-eps || b.ymax < a.ymin-eps) return 0;
    // Work near the origin to avoid area cancellation at large world offsets.
    const Point origin = a.corners[0];
    std::array<Point, 16> input, output;
    int count = 4;
    for (int k = 0; k < 4; ++k)
        input[k] = {a.corners[k].x-origin.x, a.corners[k].y-origin.y};
    for (int edge = 0; edge < 4 && count; ++edge) {
        const auto u0 = b.corners[edge], v0 = b.corners[(edge+1)%4];
        const Point u = {u0.x-origin.x,u0.y-origin.y};
        const Point v = {v0.x-origin.x,v0.y-origin.y};
        int next_count = 0;
        auto append = [&](Point p) {
            if (next_count >= static_cast<int>(output.size()))
                throw std::runtime_error("degenerate rectangle clipping exceeded vertex capacity");
            output[next_count++] = p;
        };
        auto previous = input[count-1];
        bool previous_inside = cross(u,v,previous) >= -eps;
        for (int k = 0; k < count; ++k) {
            const auto current = input[k];
            const bool current_inside = cross(u,v,current) >= -eps;
            if (current_inside != previous_inside)
                append(intersection(previous,current,u,v));
            if (current_inside) append(current);
            previous = current; previous_inside = current_inside;
        }
        count = next_count;
        input.swap(output);
    }
    double twice_area = 0;
    for (int k = 0; k < count; ++k) {
        const auto p = input[k], q = input[(k+1)%count];
        twice_area += p.x*q.y - p.y*q.x;
    }
    const double inter = std::min(std::abs(twice_area)/2, std::min(a.area,b.area));
    const double united = a.area+b.area-inter;
    return united > 1e-9 ? inter/united : 0;
}

using Boxes = py::array_t<double, py::array::c_style | py::array::forcecast>;
py::array_t<double> bev_iou_matrix(Boxes a, Boxes b) {
    for (const auto& array : {a,b}) {
        if (array.ndim() != 2 || array.shape(1) != 7)
            throw py::value_error("boxes must have shape (N, 7): x, y, z, l, w, h, yaw");
        const auto values = array.unchecked<2>();
        for (py::ssize_t i = 0; i < array.shape(0); ++i) {
            for (int k = 0; k < 7; ++k)
                if (!std::isfinite(values(i,k))) throw py::value_error("boxes must be finite");
            if (values(i,3) <= 0 || values(i,4) <= 0 || values(i,5) <= 0)
                throw py::value_error("box dimensions must be positive");
        }
    }
    const auto n = a.shape(0), m = b.shape(0);
    py::array_t<double> result({n,m});
    auto out = result.mutable_unchecked<2>();
    const double* ap = a.data(); const double* bp = b.data();
    {
        py::gil_scoped_release release;
        std::vector<Rectangle> aa, bb;
        aa.reserve(n); bb.reserve(m);
        for (py::ssize_t i = 0; i < n; ++i) aa.push_back(rectangle(ap+7*i));
        for (py::ssize_t j = 0; j < m; ++j) bb.push_back(rectangle(bp+7*j));
        for (py::ssize_t i = 0; i < n; ++i)
            for (py::ssize_t j = 0; j < m; ++j) out(i,j) = overlap(aa[i],bb[j]);
    }
    return result;
}
}  // namespace

PYBIND11_MODULE(_geometry, module) {
    module.doc() = "CPU batch BEV IoU for finite positive-size boxes, in metres and radians.";
    module.def("bev_iou_matrix", &bev_iou_matrix, py::arg("boxes_a"), py::arg("boxes_b"));
}
