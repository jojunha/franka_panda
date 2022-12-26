cmake_minimum_required(VERSION 3.0.2)

project(franka_panda)

add_compile_options(-std=c++11 -O)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    tf
)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(pybind11 REQUIRED)

set(SRC_FILES
    src/controller.cpp
    src/robotmodel.cpp
    src/trajectory.cpp
)

set(MUJOCO_PATH
    /home/kist-robot2/mujoco200
)

message("------------------------------------")
message("Mujoco Path : " ${MUJOCO_PATH})

catkin_package(
  INCLUDE_DIRS include
#   LIBRARIES franka_panda
  CATKIN_DEPENDS roscpp std_msgs 
#  DEPENDS system_lib
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${MUJOCO_PATH}/include
    /usr/include/python3.8  # for pybind11
)

link_directories(
    ${MUJOCO_PATH}/bin
    /usr/include/python3.8 
)

file(GLOB LIB_MUJOCO ${MUJOCO_PATH}/bin/libmujoco200.so)

file(GLOB UI_SRC ${MUJOCO_PATH}/include/uitools.c)

add_library(franka_lib
    ${UI_SRC}
    ${SRC_FILES}
    ${catkin_LIBRARIES}
)


# for pybind11
add_library(franka_lib_
    ${SRC_FILES}
)

add_executable(franka_panda
    src/simulate.cpp
)

target_link_libraries(franka_panda
PUBLIC
    ${catkin_LIBRARIES}
    Eigen3::Eigen
    franka_lib
    -lrbdl
    -lrbdl_urdfreader
    -lpthread

    ${MUJOCO_PATH}/bin/libglew.so
    ${MUJOCO_PATH}/bin/libglfw.so.3
    #GL
    ${LIB_MUJOCO}
    OpenGL::OpenGL
    OpenGL::EGL
    GLEW::GLEW
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}

    -lpython3.8
)


add_subdirectory(pybind11)
pybind11_add_module(controller src/controller.cpp)
target_link_libraries(controller
PUBLIC
    ${catkin_LIBRARIES}
    Eigen3::Eigen
    franka_lib_
    -lrbdl
    -lrbdl_urdfreader
    -lpthread
    -lpython3.8
    pybind11::pybind11
    ${OMPL_LIBRARIES}
)


################# For ROS ###########################
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
 )

 ################# For cmake ###########################
#install(TARGETS dualarm_exe DESTINATION bin)

#target_include_directories(dualarm_lib
#PUBLIC
#    ${CMAKE_SOURCE_DIR}/include
#)
