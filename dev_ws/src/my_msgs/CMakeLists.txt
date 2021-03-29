cmake_minimum_required(VERSION 3.5)
project(my_msgs)

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif ()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate ROS interfaces
rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/MyData.msg"
  "srv/MyService.srv"
  "srv/Goal.srv"
)

ament_package()
