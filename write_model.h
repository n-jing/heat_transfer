#ifndef WRITE_MODEL_JJ_H
#define WRITE_MODEL_JJ_H

#include "error_deal.h"
#include "copy_macro.h"
#include <cstdio>
#include <Eigen/Core>
#include <string>
#include <vector>
#include <tuple>
#include <array>
#include <iostream>
#include <numeric>


namespace Jing
{
  //! \brief write obj to path
  //! \param [in] path
  //! \return is_write_success
  template<typename T, typename U=int>
  bool WriteObj(const Eigen::Matrix<T, -1, -1> &verts,
                const Eigen::Matrix<U, -1, -1> &cells, const char* path = "demo.obj")
  {
    FILE *f_out = fopen(path, "w");
    if (!f_out)
    {
      printf("[  \033[1;31merror\033[0m  ] error in file open\n");
      return false;
    }

    fputs("# write model by program\n", f_out);
    const int VR = verts.rows();
    for (int r = 0; r < VR; ++r)
    {
      fprintf(f_out, "v %lf %lf %lf\n", verts(r, 0), verts(r, 1), verts(r, 2));
    }
    const int CR = cells.rows();
    for (int r = 0; r < CR; ++r)
    {
      fprintf(f_out, "f %d %d %d\n", cells(r, 0)+1, cells(r, 1)+1, cells(r, 2)+1);
    }

    fclose(f_out);
    return true;
  }

  template<typename T=double>
  bool AddCellAttributeToVTKModel(const char* path, const Eigen::Matrix<T, -1, 1> &attr, const char *attr_name = "attr");
  
  template<typename T=double>
  bool AddPointAttributeToVTKModel(const char* path, const Eigen::Matrix<T, -1, 1> &attr, const char *attr_name = "attr");

  //! \brief read vtk from path
  //! \param [in] path
  //! \return tuple, is_read_success, vertices, cells
  template<typename T=double, typename U=int, typename W=double>
  bool WriteTetHex(const Eigen::Matrix<T, -1, -1> &verts,
                   const Eigen::Matrix<U, -1, -1> &cells, const char* path = "demo.vtk",
                   const Eigen::Matrix<W, -1, 1> &attr = Eigen::Matrix<W, -1, 1>::Zero(0))
  {
    assert(cells.cols() == 8 || cells.cols() == 4 || cells.cols() == 3);
    FILE *f_out = fopen(path, "w");
    if (!f_out)
    {
      printf("[  \033[1;31merror\033[0m  ] error in file open\n");
      return false;
    }

    fprintf(f_out, "# vtk DataFile Version 2.0\ndemo_vtk\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS %d float\n", verts.rows());
    const int VR = verts.rows();
    for (int r = 0; r < VR; ++r)
    {
      fprintf(f_out, "%lf %lf %lf\n", verts(r, 0), verts(r, 1), verts(r, 2));
    }
    fprintf(f_out, "CELLS %d %d\n", cells.rows(), cells.rows()*(cells.cols()+1));
    const int CR = cells.rows();
    const int CC = cells.cols();
    for (int r = 0; r < CR; ++r)
    {
      fprintf(f_out, "%d", CC);
      for (int c = 0; c < CC; ++c)
      {
        fprintf(f_out, " %d", cells(r, c));
      }
      fprintf(f_out, "\n");
    }
    fprintf(f_out, "CELL_TYPES %d\n", cells.rows());
    int cell_type = 10;
    if (CC == 8)
      cell_type = 12;
    else if (CC == 3)
      cell_type = 5;
    for (int r = 0; r < CR; ++r)
    {
      fprintf(f_out, "%d\n", cell_type);
    }

    fclose(f_out);
    if (attr.size() == cells.rows())
    {
      AddCellAttributeToVTKModel(path, attr, "cell_attr");
    }
    else if (attr.size() == verts.rows())
    {
      AddPointAttributeToVTKModel(path, attr, "point_attr");
    }
    
    return true;
  }

  
  template<typename T=double>
  bool AddAttributeToVTKModel(const char* path, const Eigen::Matrix<T, -1, 1> &attr,
                              const char *type, const char *attr_name = "attr",
                              bool reset = false)
  {
    FILE *f_out = fopen(path, "a");
    if (!f_out)
    {
      printf("[  \033[1;31merror\033[0m  ] error in file open\n");
      return false;
    }
    fprintf(f_out, "%s %d\n", type, attr.size());
    fprintf(f_out, "SCALARS %s double 1\n", attr_name);
    fprintf(f_out, "LOOKUP_TABLE %s\n", attr_name);

    static T min_atrr = attr.minCoeff();
    static T max_attr = attr.maxCoeff();
    if (reset)
    {
      min_atrr = std::numeric_limits<T>::max();
      max_attr = -std::numeric_limits<T>::max();
    }
    if (attr.minCoeff() < min_atrr)
    {
      min_atrr = attr.minCoeff();
    }
    if (attr.maxCoeff() > max_attr)
    {
      max_attr = attr.maxCoeff();
    }
    printf("%lf %lf %lf %lf\n", min_atrr, max_attr, attr.minCoeff(),attr.maxCoeff());
    const int R = attr.rows();
    for (int i = 0; i < R; ++i)
      fprintf(f_out, "%lf\n", (attr(i) - min_atrr) / (max_attr - min_atrr));

    
    fprintf(f_out, "LOOKUP_TABLE %s %d\n", attr_name, 10);
    fprintf(f_out, "%lf %lf %lf %lf\n", 0/255.0,    74/255.0,  68/255.0,  255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 0/255.0,    145/255.0, 54/255.0,  255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 66/255.0,   166/255.0, 42/255.0,  255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 255/255.0,  236/255.0, 1/255.0,   255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 253/255.0,  195/255.0, 0/255.0,   255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 243/255.0,  149/255.0, 1/255.0,   255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 233/255.0,  93/255.0,  14/255.0,  255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 213/255.0,  1/255.0,   24/255.0,  255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 180/255.0,  5/255.0,   22/255.0,  255/255.0);
    fprintf(f_out, "%lf %lf %lf %lf\n", 140/255.0,  3/255.0,   49/255.0,  255/255.0);

    fclose(f_out);
    return true;
  }

  template<typename T=double>
  bool AddCellAttributeToVTKModel(const char* path, const Eigen::Matrix<T, -1, 1> &attr, const char *attr_name)
  {
    return AddAttributeToVTKModel(path, attr, "CELL_DATA", attr_name);
  }
  
  template<typename T=double>
  bool AddPointAttributeToVTKModel(const char* path, const Eigen::Matrix<T, -1, 1> &attr, const char *attr_name)
  {
    return AddAttributeToVTKModel(path, attr, "POINT_DATA", attr_name);
  }


  template<typename T=double, typename U=int>
  bool WriteModel(const Eigen::Matrix<T, -1, -1> &verts,
                  const Eigen::Matrix<U, -1, -1> &cells, const char* path)
  {
    std::string str_path(path);
    if (str_path.rfind(".obj") != std::string::npos)
    {
      return WriteObj(verts, cells, path);
    }
    else if (str_path.rfind(".vtk") != std::string::npos)
    {
      return WriteTetHex(verts, cells, path);
    }
    else
    {
      printf("[  \033[1;31merror\033[0m  ] unsupported model type. only obj and vtk are supported\n");
      return false;
    }

    return true;
  }

  template<typename T=double, typename U=int, typename W=double>
  bool WriteModel(const Eigen::Matrix<T, -1, -1> &verts,
                  const Eigen::Matrix<U, -1, -1> &cells, const char* path,
                  const Eigen::Matrix<W, -1, 1> &attr = Eigen::Matrix<W, -1, 1>::Zero(0))
  {
    return WriteTetHex<T, U, W>(verts, cells, path, attr);
  }

  template bool WriteModel<double, int>(
    const Eigen::Matrix<double, -1, -1> &verts,
    const Eigen::Matrix<int, -1, -1> &cells, const char* path);

  template bool WriteModel<float, int>(
    const Eigen::Matrix<float, -1, -1> &verts,
    const Eigen::Matrix<int, -1, -1> &cells, const char* path);

  template bool WriteModel<double, int, double>(
    const Eigen::Matrix<double, -1, -1> &verts,
    const Eigen::Matrix<int, -1, -1> &cells, const char* path,
    const Eigen::Matrix<double, -1, 1> &attr);

  template bool WriteModel<float, int, float>(
    const Eigen::Matrix<float, -1, -1> &verts,
    const Eigen::Matrix<int, -1, -1> &cells, const char* path,
    const Eigen::Matrix<float, -1, 1> &attr);

}

#endif // WRITE_MODEL_JJ_H
