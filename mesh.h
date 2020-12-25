#ifndef MESH_JJ_H
#define MESH_JJ_H

#include <Eigen/Core>
#include <set>
#include <random>
#include <iostream>

template<typename T=float, typename U=int>
Eigen::Matrix<T, -1, -1, Eigen::RowMajor> UpdateVertsData(const Eigen::Matrix<T, -1, -1> &verts, const Eigen::Matrix<T, -1, -1> &verts_norm, const Eigen::Matrix<T, -1, 1> &t, const Eigen::Matrix<int, -1, -1> &cells, const Eigen::Matrix<int, -1, -1> &cells_norm)
{
  const int verts_num = verts.rows();
  Eigen::Matrix<T, -1, -1, Eigen::RowMajor> verts_data = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>::Zero(verts_num, 7);
  verts_data.col(6) = ((t.array() - t.minCoeff())/(t.maxCoeff() - t.minCoeff())).matrix();
  verts_data.block(0, 0, verts_num, 3) = verts;

  const int cells_num = cells.rows();
  for (int c = 0; c < cells_num; ++c)
  {
    for (int i = 0; i < 3; ++i)
    {
      verts_data.block(cells(c, i), 3, 1, 3) = verts_norm.row(cells_norm(c, i));
    }
  }

  return verts_data;
}


template<typename T=float>
std::tuple<Eigen::Matrix<T, -1, 1>, std::set<int>> GenerateTemperature(int verts_num, int source_num = 10)
{
  std::default_random_engine e;
  std::uniform_int_distribution<int> source_ger(0, verts_num-1);
  std::uniform_real_distribution<T> source_temperature_ger(10, 100);

  // asset(vert_num > 10);
  std::set<int> source;
  Eigen::Matrix<T, -1, 1> temperature = Eigen::Matrix<T, -1, 1>::Zero(verts_num);

  for (int i = 0; i < source_num; ++i)
  {
    int s = source_ger(e);
    source.insert(s);
    temperature(s) = source_temperature_ger(e);
  }

  std::uniform_real_distribution<T> temperature_ger(1, 50);
  for (int v = 0; v < verts_num; ++v)
  {
    if (source.count(v))
      continue;
    temperature(v) = temperature_ger(e);
  }

  return {temperature, source};
}


#endif // MESH_JJ_H
