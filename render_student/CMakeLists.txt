cmake_minimum_required(VERSION 2.8.6)
project(render_student)


#dubug mode
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

link_directories(../core/)
include_directories(../core/)
include_directories(../third_party/matplotlib-cpp)

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(PythonLibs REQUIRED)

if(BOOST_166)
  find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)
else(BOOST_166)
  find_package(Boost 1.72.0 COMPONENTS filesystem regex system python36 numpy36)
endif(BOOST_166)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")
add_executable(render_student ${srcs})
target_link_libraries(render_student ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut mss)
