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

add_executable(steering_wheel_driver_node src/steering_wheel_driver_node.cpp)
target_link_libraries(steering_wheel_driver_node ${catkin_LIBRARIES})

add_library(steering_wheel_driver_plugin src/g29_plugin.cpp)
target_link_libraries(steering_wheel_driver_plugin ${catkin_LIBRARIES})

# CPP Execution programs
set(CPP_EXEC_NAMES steering_wheel_driver_node)
foreach(cpp_exec_names ${CPP_EXEC_NAMES})
  install(TARGETS ${cpp_exec_names}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endforeach(cpp_exec_names)

# include header files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install library
install(TARGETS steering_wheel_driver_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)