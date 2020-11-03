#include "read_model.h"
#include "write_model.h"
#include <iostream>
#include <Eigen/Sparse>
#include <set>
#include <vector>
#include <iostream>
#include <random>
#include <Eigen/SparseQR>
#include<Eigen/SparseCholesky>
#include<Eigen/SparseLU>

using namespace std;
using namespace Jing;
using namespace Eigen;

typedef double FLOAT;

int main (int argc, char *argv[])
{
  bool read_success = false;
  Eigen::Matrix<FLOAT, -1, -1> verts;
  Eigen::Matrix<int, -1, -1> cells;
  tie(read_success, verts, cells) = ReadModel<FLOAT, int>(argv[1]);
  if (!read_success)
  {
    cerr << "[  \033[1;31merror\033[0m  ] " << "error in read model!" << endl;
    return -1;
  }


  const int vert_num = verts.rows();
  default_random_engine e;
  uniform_int_distribution<int> source_ger(0, vert_num);
  uniform_real_distribution<FLOAT> source_temperature_ger(10, 100);

  // asset(vert_num > 10);
  set<int> source;
  const int source_num = 10;

  Matrix<FLOAT, -1, 1> temperature = Matrix<FLOAT, -1, 1>::Zero(vert_num);
  for (int i = 0; i < source_num; ++i)
  {
    int s = source_ger(e);
    source.insert(s);
    temperature(s) = source_temperature_ger(e);
  }

  vector<int> vert_lap_id(vert_num, 0);
  vector<int> lap_vert_id(vert_num, 0);
  iota(lap_vert_id.begin(), lap_vert_id.end(), 0);
  int vert_count = 0;
  for (int i = 0; i < vert_num; ++i)
  {
    if (source.count(i))
      continue;
    vert_lap_id[i] = vert_count;
    lap_vert_id[vert_count] = i;
    vert_count++;
  }

  Matrix<FLOAT, -1, 1> b = Matrix<FLOAT, -1, 1>::Zero(vert_count);
  SparseMatrix<FLOAT> laplace(vert_count, vert_count);
  {
    vector<set<int>> verts_neighbor(verts.rows());
    const int R = cells.rows();
    const int C = cells.cols();
    for (int r = 0; r < R; ++r)
    {
      for (int c = 0; c < C; ++c)
      {
        for (int k = 0; k < C; ++k)
        {
          if (k == c)
            continue;
          verts_neighbor[cells(r, c)].insert(cells(r, k));
        }
      }
    }

    vector<Triplet<FLOAT>> laplace_element;
    for (int v = 0; v < verts.rows(); ++v)
    {
      if (source.count(v))
        continue;
      laplace_element.emplace_back(vert_lap_id[v], vert_lap_id[v], -1.0);
      for (auto n : verts_neighbor[v])
      {
        if (source.count(n))
        {
          b(vert_lap_id[v]) -= temperature(n) / verts_neighbor[v].size();
        }
        else
        {
          laplace_element.emplace_back(vert_lap_id[v], vert_lap_id[n], 1.0 / verts_neighbor[v].size());
        }
      }
    }
    laplace.setFromTriplets(laplace_element.begin(), laplace_element.end());
  }

  SparseLU<SparseMatrix<FLOAT>> solver;
  solver.compute(laplace);
  Matrix<FLOAT, -1, 1> t = solver.solve(b);

  Matrix<FLOAT, -1, 1> balance = Matrix<FLOAT, -1, 1>::Zero(vert_num);
  for (int i = 0; i < vert_num; ++i)
  {
    if (source.count(i))
      balance(i) = temperature(i);
    else
      balance(i) = t[vert_lap_id[i]];
  }

  WriteModel<double, int, double>(verts, cells, "balance.vtk", balance);

  return 0;
}
