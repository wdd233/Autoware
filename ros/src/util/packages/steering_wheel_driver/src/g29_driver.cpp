#include <steering_wheel_driver/g29_driver.h>

G29Driver::G29Driver(ros::NodeHandle nh, ros::NodeHandle pnh) {
  nh_ = nh;
  pnh_ = pnh;
  status_ = boost::none;
  joystick_id_ = 0;
  setup();
}

G29Driver::~G29Driver() {}

void G29Driver::run() { return; }

void G29Driver::publishCmd() {
  if (!status_) {
    return;
  }
  return;
}

int G29Driver::ffUpdate(SDL_Joystick *joystick, unsigned short center) {
  SDL_Haptic *haptic;
  SDL_HapticEffect effect;
  int effect_id;

  // Open the device
  haptic = SDL_HapticOpenFromJoystick(joystick);
  if (haptic == NULL)
    return -1; // Most likely joystick isn't haptic

  // See if it can do sine waves
  if ((SDL_HapticQuery(haptic) & SDL_HAPTIC_SPRING) == 0) {
    SDL_HapticClose(haptic); // No spring effect
    return -2;
  }

  // Create the effect
  memset(&effect, 0, sizeof(SDL_HapticEffect)); // 0 is safe default
  effect.type = SDL_HAPTIC_SPRING;
  effect.condition.length = 30;                           // 30ms long
  effect.condition.delay = 0;                             // no delay
  effect.condition.center[0] = (int16_t)(center - 32768); // NEED CONVERSION!!!

  // Upload the effect
  effect_id = SDL_HapticNewEffect(haptic, &effect);

  // Test the effect
  SDL_HapticRunEffect(haptic, effect_id, 1);
  SDL_Delay(5000); // Wait for the effect to finish

  // We destroy the effect, although closing the device also does this
  SDL_HapticDestroyEffect(haptic, effect_id);

  // Close the device
  SDL_HapticClose(haptic);

  return 0; // Success
}

void G29Driver::vehicleStatusFeedback(autoware_msgs::VehicleStatus msg) {
  status_ = msg;
  return;
}

void G29Driver::changeSteeringAngle(double target_angle){
  return;
}

void G29Driver::addForce(double target_force){
  return;
}

void G29Driver::setup() {
  if (SDL_Init(SDL_INIT_JOYSTICK) < 0){
    ROS_ERROR_STREAM("Error initializing SDL!");
    std::exit(0);
  }
  joy_=SDL_JoystickOpen(joystick_id_);
  if (joy_) {
    ROS_INFO_STREAM("connect to " << SDL_JoystickNameForIndex(0));
  }
  else{
    ROS_ERROR_STREAM("failed to open G29");
    std::exit(0);
  }
  hid_init();
  return;
}