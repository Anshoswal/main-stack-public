from visualization_msgs.msg import Marker, MarkerArray
from dv_msgs.msg import Track, Cone, ObservationRangeBearing, SingleRangeBearingObservation

def send_to_SLAM(self, thetas, ranges, colors):
        ObsMsg = ObservationRangeBearing()
        ObsMsg.header.frame_id = 'bot'
        ObsMsg.header.stamp = self.get_clock().now().to_msg()

        # The sensor pose on the robot coordinates frame: (from settings.json)
        ObsMsg.sensor_pose_on_robot.position.x = float(0)
        ObsMsg.sensor_pose_on_robot.position.y = float(0)
        ObsMsg.sensor_pose_on_robot.position.z = float(0)
        ObsMsg.sensor_pose_on_robot.orientation.w = float(0)
        ObsMsg.sensor_pose_on_robot.orientation.x = float(0)
        ObsMsg.sensor_pose_on_robot.orientation.y = float(0)
        ObsMsg.sensor_pose_on_robot.orientation.z = float(0)
        # Sensor characteristics:
        ObsMsg.min_sensor_distance = float(0)
        ObsMsg.max_sensor_distance = float(30)
        # Typical sensor noise:
        ObsMsg.sensor_std_range = float(0)
        ObsMsg.sensor_std_pitch = float(0)
        ObsMsg.sensor_std_yaw = float(0)

        sensed_data = []

        ConesSeen = Track()
        cones = []

        carLocation = Cone()

        for i in range(len(thetas)):
            single = SingleRangeBearingObservation()
            if(ranges[i] > 25):
                continue
            single.range = ranges[i]
            single.yaw = thetas[i] * pi / 180.0 
            single.pitch = float(0)
            single.id = -1 


            cone_P = Cone()
            # represents range, pitch, yaw
            cone_P.location.x =  ranges[i]
            cone_P.location.y = thetas[i] * pi / 180.0 
            cone_P.location.z = float(0)

            if (colors[i] == 0):
                cone_P.color = 0
            elif colors[i] == 1:
                cone_P.color = 2 # Big Orange
            elif colors[i] == 3 or colors[i] == 4:
                cone_P.color = 1
            elif colors[i] == 2:
                cone_P.color = 3 # Small Orange

            cones.append(cone_P)

            sensed_data.append(single)

        ConesSeen.track = cones
        self.get_logger().info(f"numuber of sensed data is = {len(sensed_data)} and length of thetas = {len(thetas)}")

        if len(sensed_data) > 0:
            self.ConesSeenPerception.publish(ConesSeen)

            deptharray = MarkerArray()
            for i in range(len(sensed_data)):
                marker = Marker()
                marker.header.frame_id = "base_footprint"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "my_namespace"
                marker.id = i
                marker.type = 1
                marker.action = 0
                marker.pose.position.x = (ConesSeen.track[i].location.x)*cos(ConesSeen.track[i].location.y)
                marker.pose.position.y = (ConesSeen.track[i].location.x)*sin(ConesSeen.track[i].location.y)
                marker.pose.position.z = 0.0 # ConesSeen.track[i].location.z
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                # marker.pose.orientation.w = 1.0
                marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                marker.color.a = 1.0
                if ConesSeen.track[i].color == 0:
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                elif ConesSeen.track[i].color == 1:
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif ConesSeen.track[i].color == 3: # 3 = small orange
                    marker.color.r = 0.945
                    marker.color.g = 0.353
                    marker.color.b = 0.134
                elif ConesSeen.track[i].color == 2: # 2 = big orange
                    marker.color.r = 1.0
                    marker.color.g = 0.58
                    marker.color.b = 0.44
                elif ConesSeen.track[i].color == 4:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                deptharray.markers.append(marker)

            self.viz_depths.publish(deptharray)
