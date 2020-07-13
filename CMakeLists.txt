# LastEditTime: 2020-06-04
# 1.cmake verson 
cmake_minimum_required(VERSION 3.2)

# 2.project name
PROJECT(MSTdepth)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -W -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -w") 

# 3.head file path
INCLUDE_DIRECTORIES(include)

# 4.source file directory
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

# 5.set environment variable
SET(SRC_ENV ${DIR_SRCS})

#6.add executable file
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_ENV})

#--------------- Eigen3----------------#
find_package(Eigen3)
#special eigen3 path (using it when can't find Eigen3 automatically) 
#set(EIGEN3_INCLUDE_DIR "/home/liminghao/eigen")
# directory of Eigen3 headers
include_directories(${EIGEN3_INCLUDE_DIR})
#-------------------------------------------#

#--------------- OpenCV----------------#
#special opencv path (using it when can't find OpenCV automatically) 
#set( CMAKE_PREFIX_PATH "/home/liminghao/local" )
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#-------------------------------------------#

#7.add link
TARGET_LINK_LIBRARIES(${PROJECT_NAME} )
