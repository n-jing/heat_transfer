#ifndef INIT_WINDOW_JJ_H
#define INIT_WINDOW_JJ_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "camera.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
CameraMovement processInput(GLFWwindow *window, float deltaTime);

unsigned int loadTexture(const char *path);

GLFWwindow* InitWindow(unsigned int SCR_WIDTH=800, unsigned int SCR_HEIGHT=600, const char *name = "Demo");

#endif // INIT_WINDOW_JJ_H
