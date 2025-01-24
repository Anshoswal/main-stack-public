cmake_minimum_required(VERSION 3.5)
project(dv_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)


# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObservationRangeBearing.msg"
  "msg/Odometry.msg"
  "msg/SingleRangeBearingObservation.msg"
  "msg/Cone.msg"
  "msg/Track.msg"
  "msg/Pose.msg"
  "msg/MarkerArray.msg"
  "msg/ControlCommand.msg"
  "msg/TwistWithCovarianceStamped.msg"
  "msg/TransformStamped.msg"
  "msg/AI2LOGDynamics2.msg"
  "msg/AI2VCUBrake.msg"
  "msg/AI2VCUDriveF.msg"
  "msg/AI2VCUDriveR.msg"
  "msg/AI2VCUStatus.msg"
  "msg/AI2VCUSteer.msg"
  "msg/VCUSTATUS.msg"
  "msg/VCU2AIBrake.msg"
  "msg/VCU2AIDriveF.msg"
  "msg/VCU2AIDriveR.msg"
  "msg/VCU2AISpeeds.msg"
  "msg/VCU2AIStatus.msg"
  "msg/VCU2AISteer.msg"
  "msg/VCU2AIWheelcounts.msg"
  "msg/VCU2LOGDynamics1.msg"
  "msg/VCU2LOGStatus.msg"
  "msg/PointArray.msg"
  "srv/Reset.srv"
  DEPENDENCIES geometry_msgs visualization_msgs std_msgs 
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
