#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "init_window.h"
#include "mesh.h"


#include "read_model.h"
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include<Eigen/SparseLU>


using namespace Eigen;
using namespace std;
using namespace Jing;

using FLOAT = float;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 70.0f));

// lighting
glm::vec3 lightPos(70.2f, 71.0f, 72.0f);

int main(int argc, char *argv[])
{
// settings
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 600;

  GLFWwindow* window = InitWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL");
  glEnable(GL_PROGRAM_POINT_SIZE);
  
  // build and compile our shader zprogram
  // ------------------------------------
  Shader lightingShader; //("../shader/demo.vs", "../shader/demo.fs");
  const char *vs_path[] = {"../shader/demo.vs"};
  const char *fs_path[] = {"../shader/version.frag", "../shader/transform_rainbow.frag", "../shader/demo.fs"};

  Shader pointShader("../shader/point.vs", "../shader/point.fs");
  
  lightingShader.CompileVertexShader(1, vs_path);
  lightingShader.CompileFragmentShader(3, fs_path);
  lightingShader.CompileShader();
  
  const char *path = argv[1];
  bool read_success = false;
  Eigen::Matrix<FLOAT, -1, -1> verts;
  Eigen::Matrix<int, -1, -1> cells;
  Eigen::Matrix<FLOAT, -1, -1> verts_norm;
  Eigen::Matrix<int, -1, -1> cells_norm;
  tie(read_success, verts, verts_norm, cells, cells_norm) = ReadObjWithNorm<FLOAT, int>(path);
  if (!read_success)
  {
    cerr << "[  \033[1;31merror\033[0m  ] " << "error in read model!" << endl;
    return -1;
  }
  const int verts_num = verts.rows();

  
  Eigen::Matrix<FLOAT, -1, 1> temperature;
  std::set<int> source;
  tie(temperature, source) = GenerateTemperature<FLOAT>(verts_num, 3);
  Matrix<FLOAT, -1, -1, RowMajor> verts_data = UpdateVertsData<FLOAT, int>(verts, verts_norm, temperature, cells, cells_norm);
  Matrix<int, -1, -1, RowMajor> verts_cell = cells;
  Matrix<FLOAT, -1, -1, RowMajor> verts_points(source.size(), 3);
  int s_count = 0;
  for (auto s : source)
  {
    verts_points.row(s_count++) = verts.row(s);
  }

// first, configure the cube's VAO (and VBO)
  unsigned int VBO, VAO, VAO_P, EBO;
  glGenVertexArrays(1, &VAO);
  glGenVertexArrays(1, &VAO_P);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, verts_data.size() * sizeof(FLOAT), verts_data.data(), GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, verts_cell.size() * sizeof(int), verts_cell.data(), GL_STATIC_DRAW);
  
  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(FLOAT), (void*)0);
  glEnableVertexAttribArray(0);
  // normal attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(FLOAT), (void*)(3 * sizeof(FLOAT)));
  glEnableVertexAttribArray(1);
  // normal attribute
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(FLOAT), (void*)(6 * sizeof(FLOAT)));
  glEnableVertexAttribArray(2);


  glBindVertexArray(VAO_P);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, verts_points.size() * sizeof(FLOAT), verts_points.data(), GL_DYNAMIC_DRAW);
  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(FLOAT), (void*)0);
  glEnableVertexAttribArray(0);

  // render loop
  // -----------
  FLOAT lastFrame = 0.0f;
  while (!glfwWindowShouldClose(window))
  {
    // per-frame time logic
    // --------------------
    FLOAT currentFrame = glfwGetTime();
    FLOAT deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    // -----
    CameraMovement k = processInput(window, deltaTime);
    if (k == CameraMovement::SPACE)
    {
      vector<int> vert_lap_id(verts_num, 0);
      int vert_count = 0;
      for (int i = 0; i < verts_num; ++i)
      {
        if (source.count(i))
          continue;
        vert_lap_id[i] = vert_count;
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
      
      Matrix<FLOAT, -1, 1> balance = Matrix<FLOAT, -1, 1>::Zero(verts_num);
      for (int i = 0; i < verts_num; ++i)
      {
        if (source.count(i))
          balance(i) = temperature(i);
        else
          balance(i) = t[vert_lap_id[i]];
      }

      verts_data.col(6) = ((balance.array() - balance.minCoeff())/(balance.maxCoeff() - balance.minCoeff())).matrix();
    }
    
    // render
    // ------
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // be sure to activate shader when setting uniforms/drawing objects
    lightingShader.use();
    lightingShader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
    lightingShader.setVec3("lightPos", lightPos);
    lightingShader.setVec3("viewPos", camera.Position);

    // view/projection transformations
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = camera.GetViewMatrix();
    lightingShader.setMat4("projection", projection);
    lightingShader.setMat4("view", view);

    // world transformation
    glm::mat4 model = glm::mat4(1.0f);
    float angle = 10*glfwGetTime();
    model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 1.0f, 0.0f));
    lightingShader.setMat4("model", model);

    // render the cube
    glBufferData(GL_ARRAY_BUFFER, verts_data.size() * sizeof(FLOAT), verts_data.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, verts_cell.size(), GL_UNSIGNED_INT, 0);

    pointShader.use();
    pointShader.setMat4("projection", projection);
    pointShader.setMat4("view", view);
    pointShader.setMat4("model", model);

    
    glBufferData(GL_ARRAY_BUFFER, verts_points.size() * sizeof(FLOAT), verts_points.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(VAO_P);
    // glVertexPointer(2, GL_FLOAT, 0, VAO_P);
    glDrawArrays(GL_POINTS, 0, verts_data.rows());
    
    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  
  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(1, &VAO);
  glDeleteVertexArrays(1, &VAO_P);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}
