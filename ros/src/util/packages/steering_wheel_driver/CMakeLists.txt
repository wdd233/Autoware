cmake_minimum_required(VERSION 2.8.3)
project(steering_wheel_driver)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  roscpp
  pluginlib
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS autoware_msgs roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(g29_driver_node src/g29_driver_node.cpp src/g29_driver.cpp)
target_link_libraries(g29_driver_node ${catkin_LIBRARIES})

# CPP Execution programs
set(CPP_EXEC_NAMES g29_driver_node)
foreach(cpp_exec_names ${CPP_EXEC_NAMES})
  install(TARGETS ${cpp_exec_names}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endforeach(cpp_exec_names)

# include header files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

foreach(dir launch)
  install(DIRECTORY ${dir}/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)