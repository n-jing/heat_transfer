ADD_LIBRARY(glad STATIC ${CMAKE_SOURCE_DIR}/extern/glad/src/glad.c)
TARGET_INCLUDE_DIRECTORIES(glad PUBLIC ${CMAKE_SOURCE_DIR}/extern/glad/include ${CMAKE_SOURCE_DIR}/extern/glm)

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
  include_directories(${GLEW_INCLUDE_DIRS})
  link_libraries(${GLEW_LIBRARIES})
endif()


find_package(assimp REQUIRED)

FILE(GLOB source *.cpp)
ADD_EXECUTABLE(display ${source})
TARGET_INCLUDE_DIRECTORIES(display PUBLIC ${CMAKE_SOURCE_DIR}/extern/glad/include ${CMAKE_SOURCE_DIR})
TARGET_LINK_LIBRARIES(display glad dl GL GLU glut glfw ${ASSIMP_LIBRARIES})
SET_TARGET_PROPERTIES(display
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ADD_SUBDIRECTORY(doc EXCLUDE_FROM_ALL)
