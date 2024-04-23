cmake_minimum_required(VERSION 2.8.6)
project(mss)

#SET(CMAKE_BUILD_TYPE Release CACHE STRING
#	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

#dubug mode
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(DART REQUIRED COMPONENTS utils-urdf gui collision-bullet CONFIG)
find_package(TinyXML REQUIRED)

if(BOOST_166)
  find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)
else(BOOST_166)
  find_package(Boost 1.72.0 COMPONENTS filesystem regex system python36 numpy36)
endif(BOOST_166)

include_directories(${DART_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")

add_library(mss ${srcs})
target_link_libraries(mss ${DART_LIBRARIES} ${TinyXML_LIBRARIES})
