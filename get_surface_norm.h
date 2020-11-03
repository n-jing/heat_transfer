#ifndef GET_SURFACE_NORM_JJ_H
#define GET_SURFACE_NORM_JJ_H



#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Jing
{
  template<typename T=float, typename U=int>
  Eigen::Matrix<T, -1, -1> GetSurfaceNorm(const Eigen::Matrix<T, -1, -1> &verts, const Eigen::Matrix<U, -1, -1> &cells)
  {
    using matrix_vec = std::vector<Eigen::Matrix<T, -1, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, -1, 1>>>;
    assert(verts.cols() == 3 && cells.cols() == 3);
    const int verts_num = verts.rows();
    std::vector<matrix_vec, Eigen::aligned_allocator<matrix_vec>> norm_vec(verts_num);
    const int cells_num = cells.rows();
    const int cells_ele = cells.cols();
    for (int c = 0; c < cells_num; ++c)
    {
      Eigen::Matrix<T, 3, 1> ab = verts.row(cells(c, 1)) - verts.row(cells(c, 0));
      Eigen::Matrix<T, 3, 1> ac = verts.row(cells(c, 2)) - verts.row(cells(c, 0));
      Eigen::Matrix<T, 3, 1> n = ab.cross(ac).normalized();
      for (int e = 0; e < cells_ele; ++e)
        norm_vec[cells(c, e)].push_back(n);
    }

    Eigen::Matrix<T, -1, -1> surface_norm(verts_num, verts.cols());
    for (int i = 0; i < verts_num; ++i)
    {
      Eigen::Matrix<T, -1, 1> vn = Eigen::Matrix<T, -1, 1>::Zero(3);
      for (auto &ni : norm_vec[i])
      {
        vn += ni;
      }
      vn.normalize();
      surface_norm.row(i) = vn;
    }

    return surface_norm;
  }

}



#endif // GET_SURFACE_NORM_JJ_H
