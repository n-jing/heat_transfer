#include "read_model.h"
#include "write_model.h"
#include <iostream>
#include <Eigen/Sparse>
#include <set>
#include <vector>
#include <iostream>
#include <random>
#include <Eigen/SparseQR>

using namespace std;
using namespace Jing;
using namespace Eigen;


int main (int argc, char *argv[])
{
  bool read_success = false;
  Eigen::Matrix<double, -1, -1> verts;
  Eigen::Matrix<int, -1, -1> cells;
  tie(read_success, verts, cells) = ReadModel<double, int>(argv[1]);
  if (!read_success)
  {
    cerr << "[  \033[1;31merror\033[0m  ] " << "error in read model!" << endl;
    return -1;
  }

  SparseMatrix<double> laplace(verts.rows(), verts.rows());
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

    vector<Triplet<double>> laplace_element;
    for (int v = 0; v < verts.rows(); ++v)
    {
      laplace_element.emplace_back(v, v, -1.0);
      for (auto n : verts_neighbor[v])
        laplace_element.emplace_back(v, n, 1.0 / verts_neighbor[v].size());
    }
    laplace.setFromTriplets(laplace_element.begin(), laplace_element.end());
  }

  const int vert_num = verts.rows();
  default_random_engine e;
  uniform_int_distribution<int> source_ger(0, vert_num);
  uniform_real_distribution<double> source_temperature_ger(90, 100);
  uniform_real_distribution<double> temperature_ger(0, 10);

  // assert(vert_num > 10);
  set<int> source;
  const int source_num = 10;
  vector<Triplet<double>> J_ele; // constrian matrix
  for (int i = 0; i < source_num; ++i)
  {
    int s = source_ger(e);
    source.insert(s);
    J_ele.emplace_back(s, s, 1);
  }

  SparseMatrix<double> J(vert_num, vert_num);
  J.setFromTriplets(J_ele.begin(), J_ele.end());
  Matrix<double, -1, 1> constrian(vert_num); // constrain
  Matrix<double, -1, 1> temperature(vert_num);
  for (int i = 0; i < vert_num; ++i)
  {
    if (source.count(i))
    {
      temperature(i) = source_temperature_ger(e);
      constrian(i) = temperature(i);
    } 
    else
    {
      temperature(i) = temperature_ger(e);
      constrian(i) = 0;
    }
      
  }

  WriteModel<double, int, double>(verts, cells, "origin.vtk", temperature);

  Matrix<double, -1, 1> JTconstrain = J.transpose() * constrian;
  SparseMatrix<double> JTJ = J.transpose() * J;
  SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr;
  qr.compute(JTJ);

  const double h = 1e-2;
  const double lambda = 1;
  for (int i = 0; i < 3000; ++i)
  {
    printf("[   \033[1;37mlog\033[0m   ] **********************frame %d***********************\n", i);
    temperature += lambda * h * laplace * temperature;
    Matrix<double, -1, 1> b = JTconstrain - JTJ * temperature;
    Matrix<double, -1, 1> q = qr.solve(b);
    temperature += q;
    string str = "frame_" + to_string(i) + ".vtk";
    WriteModel<double, int, double>(verts, cells, str.c_str(), temperature);
  }

  return 0;
}
