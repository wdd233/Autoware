// headers for ros
#include <ros/ros.h>
#include <steering_wheel_driver/g29_driver.h>

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "g29_driver_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  g29_driver driver(nh,pnh);
  ros::spin();
  return 0;
}