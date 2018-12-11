/*
 *  Copyright (c) 2017, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PACMOD_INTERFACE_H
#define PACMOD_INTERFACE_H

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>

#include <autoware_msgs/VehicleCmd.h>
#include <autoware_msgs/VehicleStatus.h>

#include <pacmod_msgs/SystemCmdBool.h>
#include <pacmod_msgs/SystemCmdInt.h>
#include <pacmod_msgs/SystemCmdFloat.h>
#include <pacmod_msgs/SteerSystemCmd.h>
#include <pacmod_msgs/SystemRptInt.h>
#include <pacmod_msgs/SystemRptFloat.h>
#include <pacmod_msgs/VehicleSpeedRpt.h>

#include "pid_controller.h"

const static double WHEEL_BASE = 2.79;

class PacmodInterface
{
public:
  PacmodInterface();
  ~PacmodInterface();

  void run();

private:
  // typedefs
  typedef message_filters::sync_policies::ApproximateTime<pacmod_msgs::VehicleSpeedRpt, pacmod_msgs::SystemRptFloat>
      PacmodTwistSyncPolicy;

  // node handles
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  // subscribers from autoware
  ros::Subscriber vehicle_cmd_sub_;
  ros::Subscriber engage_cmd_sub_;

  // subscribers from pacmod
  ros::Subscriber pacmod_enabled_sub_;
  message_filters::Subscriber<pacmod_msgs::VehicleSpeedRpt>* pacmod_speed_sub_;
  message_filters::Subscriber<pacmod_msgs::SystemRptFloat>* pacmod_steer_sub_;
  message_filters::Synchronizer<PacmodTwistSyncPolicy>* pacmod_twist_sync_;

  // publishers to autoware
  ros::Publisher vehicle_status_pub_;
  ros::Publisher current_twist_pub_;

  // publishers to pacmod
  ros::Publisher pacmod_steer_pub_;
  ros::Publisher pacmod_accel_pub_;
  ros::Publisher pacmod_brake_pub_;
  ros::Publisher pacmod_shift_pub_;
  ros::Publisher pacmod_turn_pub_;
  // ros::Publisher pacmod_headlight_pub_;
  // ros::Publisher pacmod_horn_pub_;
  // ros::Publisher pacmod_wiper_pub_;

  // rosparams
  double loop_rate_;
  double accel_kp_, accel_ki_, accel_kd_;
  double brake_kp_, brake_ki_, brake_kd_;
  double accel_max_, brake_max_;
  double brake_deadband_;
  double rotation_rate_;

  // variables
  bool engage_cmd_, prev_engage_cmd_;
  bool engage_state_, prev_engage_state_;

  bool enable_;
  bool ignore_overrides_;
  bool clear_override_;
  bool clear_faults_;

  bool init_vehicle_cmd_;
  double current_speed_, current_steer_;
  geometry_msgs::TwistStamped current_twist_;
  autoware_msgs::VehicleCmd vehicle_cmd_;

  pacmod_msgs::SteerSystemCmd pacmod_steer_;
  pacmod_msgs::SystemCmdFloat pacmod_accel_;
  pacmod_msgs::SystemCmdFloat pacmod_brake_;
  pacmod_msgs::SystemCmdInt pacmod_shift_;
  pacmod_msgs::SystemCmdInt pacmod_turn_;

  ros::Rate* rate_;

  PIDController accel_pid_;
  PIDController brake_pid_;

  // callbacks from autoware
  void callbackVehicleCmd(const autoware_msgs::VehicleCmd::ConstPtr& msg);
  void callbackEngage(const std_msgs::Bool::ConstPtr& msg);

  // callbacks from pacmod
  void callbackPacmodEnabled(const std_msgs::Bool::ConstPtr& msg);
  void callbackPacmodTwist(const pacmod_msgs::VehicleSpeedRpt::ConstPtr& speed,
                           const pacmod_msgs::SystemRptFloat::ConstPtr& steer);

  // functions
  bool checkInitialized();
  void updateOverride();
  void publishPacmodSteer(const autoware_msgs::VehicleCmd& msg);
  void publishPacmodAccel(const autoware_msgs::VehicleCmd& msg);
  void publishPacmodBrake(const autoware_msgs::VehicleCmd& msg);
  void publishPacmodShift(const autoware_msgs::VehicleCmd& msg);
  void publishPacmodTurn(const autoware_msgs::VehicleCmd& msg);
};

#endif  // PACMOD_INTERFACE_H
