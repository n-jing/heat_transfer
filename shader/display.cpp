#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "init_window.h"
#include "mesh.h"
#include "init_buffer.h"

#include "read_model.h"
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include<Eigen/SparseLU>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

using namespace Eigen;
using namespace std;
using namespace Jing;

using FLOAT = float;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 10.0f));

// lighting
glm::vec3 lightPos(70.2f, 71.0f, 72.0f);

constexpr int MAX_FRAME = 400;
GLbyte *frame[MAX_FRAME];
int frame_count = 0;

int main(int argc, char *argv[])
{
// settings
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 600;

  GLFWwindow* window = InitWindow(SCR_WIDTH, SCR_HEIGHT, "Heat Transfer");
  glEnable(GL_PROGRAM_POINT_SIZE);
  
  // build and compile our shader zprogram
  // ------------------------------------
  Shader lightingShader; //("../shader/demo.vs", "../shader/demo.fs");
  const char *vs_path[] = {"../shader/demo.vs"};
  const char *fs_path[] = {"../shader/version.frag", "../shader/gnuplot.frag", "../shader/demo.fs"};

  Shader pointShader("../shader/point.vs", "../shader/point.fs");
  Shader planeShader("../shader/plane.vs", "../shader/plane.fs");
  
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


  int source_num = atoi(argv[2]);
  Eigen::Matrix<FLOAT, -1, 1> temperature;
  std::set<int> source;
  tie(temperature, source) = GenerateTemperature<FLOAT>(verts_num, source_num);
  Matrix<FLOAT, -1, -1, RowMajor> verts_data = UpdateVertsData<FLOAT, int>(verts, verts_norm, temperature, cells, cells_norm);
  Matrix<int, -1, -1, RowMajor> verts_cell = cells;
  Matrix<FLOAT, -1, -1, RowMajor> verts_points(source.size(), 3);
  int s_count = 0;
  for (auto s : source)
  {
    verts_points.row(s_count++) = verts.row(s);
  }

// first, configure the cube's VAO (and VBO)
  unsigned int VBO, VAO, VAO_P, VBO_P, EBO;
  glGenVertexArrays(1, &VAO);
  glGenVertexArrays(1, &VAO_P);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glGenBuffers(1, &VBO_P);

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
  glBindVertexArray(0);


  glBindVertexArray(VAO_P);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_P);
  glBufferData(GL_ARRAY_BUFFER, verts_points.size() * sizeof(FLOAT), verts_points.data(), GL_DYNAMIC_DRAW);
  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(FLOAT), (void*)0);
  glEnableVertexAttribArray(0);
  // glBindBuffer(GL_ARRAY_BUFFER, 0);
  // glBindVertexArray(0);
  
  float level = -2.0;
  float plane_vertices[] = {
    // positions          // colors           // texture coords
    100.5f,  level, 100.0f,   0.0f, 1.0f, 0.0f,   100.0f, 100.0f, // top right
    100.5f, level, -100.0f,   0.0f, 1.0f, 0.0f,   100.0f, -100.0f, // bottom right
    -100.5f, level, -100.0f,   0.0f, 1.0f, 0.0f,   -100.0f, -100.0f, // bottom left
    -100.5f,  level, 100.0f,   0.0f, 1.0f, 0.0f,   -100.0f, 100.0f  // top left 
  };
  unsigned int plane_indices[] = {  
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
  };

  unsigned int planeVBO, planeVAO, planeEBO;
  glGenVertexArrays(1, &planeVAO);
  glGenBuffers(1, &planeVBO);
  glGenBuffers(1, &planeEBO);

  glBindVertexArray(planeVAO);
  glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(plane_vertices), plane_vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(plane_indices), plane_indices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(0));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3*sizeof(float)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6*sizeof(float)));
  glEnableVertexAttribArray(2);

  unsigned int texture = loadTexture("../grid.png");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

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
    // if (k == CameraMovement::SPACE)
    if (frame_count == 200)
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
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
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
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, verts_data.size() * sizeof(FLOAT), verts_data.data(), GL_DYNAMIC_DRAW);
    glDrawElements(GL_TRIANGLES, verts_cell.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    pointShader.use();
    pointShader.setMat4("projection", projection);
    pointShader.setMat4("view", view);
    pointShader.setMat4("model", model);
    glBindVertexArray(VAO_P);
    glBufferData(GL_ARRAY_BUFFER, verts_points.size() * sizeof(FLOAT), verts_points.data(), GL_DYNAMIC_DRAW);
    // glVertexPointer(2, GL_FLOAT, 0, VAO_P);
    glDrawArrays(GL_POINTS, 0, verts_data.rows());
    glBindVertexArray(0);
    
    
    //draw plane
    planeShader.use();
    planeShader.setMat4("projection", projection);
    planeShader.setMat4("view", view);
    planeShader.setMat4("model", glm::mat4(1.0f));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glBindVertexArray(planeVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();

    double fps = 30.0;
    double time_current = glfwGetTime();
    double time_accurate = frame_count / 1.0 / fps;
    double time_delta = time_accurate - time_current;
    time_delta = time_delta > 0 ? time_delta : 0;
    if(true)
      printf("frame_count:%d time_accurate:%lf time_current:%lf time_delta:%lf\n", frame_count, time_accurate, time_current, time_delta);
    frame[frame_count] = new GLbyte[SCR_WIDTH * SCR_HEIGHT * 3];
    glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, frame[frame_count++]);
    usleep(time_delta * 1000000);
    if (frame_count >= MAX_FRAME)
      break;
  }

  FILE *out = fopen("raw_video", "wb");
  for(int i = 0; i < frame_count; i++)
    fwrite(frame[i], SCR_WIDTH * SCR_HEIGHT * 3, 1, out);

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(1, &VAO);
  glDeleteVertexArrays(1, &VAO_P);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &VBO_P);
  glDeleteBuffers(1, &EBO);

  glDeleteVertexArrays(1, &planeVAO);
  glDeleteBuffers(1, &planeVBO);
  glDeleteBuffers(1, &planeEBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}
