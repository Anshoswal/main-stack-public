#include "rclcpp/rclcpp.hpp"         // Required for ROS 2 functionality
#include "std_msgs/msg/string.hpp"   // For String message type

class SlamNode : public rclcpp::Node {
public:
    SlamNode() : Node("slam_node") {
        // subscribers here 
        perception_subscriber_topic_ = "perception_topic";
        perception_subscription_ = this->create_subscription<std_msgs::msg::String>(
            perception_subscriber_topic_, 10,
            std::bind(&SlamNode::slam_callback, this, std::placeholders::_1)
        );
        
        // publishers here
        slam_publisher_topic = "slam_topic";
        slam_publisher = this->create_publisher<std_msgs::msg::String>(slam_publisher_topic, 10);
    }

private:
    void slam_callback(const std_msgs::msg::String::SharedPtr msg) {
        // Algorithm function calls will be made here
        
    }

    void send_to_planner(const std_msgs::msg::String &msg) {
        // Function to publish messages to VCU
    }

    // Subscriber and Publisher objects
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr perception_subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr slam_publisher;

    // Topic names
    std::string perception_subscriber_topic_;
    std::string slam_publisher_topic;
};

// Main function
int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
