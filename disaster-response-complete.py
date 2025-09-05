import threading
import queue
import asyncio
import time
import logging
import json
import math
import copy
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from enum import Enum
from collections import deque
import warnings
import random

# Core libraries
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import heapq

# MAVLink communication
try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    print("Warning: PyMAVLink not available, using mock flight controller")

# Optional dependencies with fallbacks
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available, using mock detection")

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: RealSense SDK not available, using mock depth data")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available, using basic point cloud processing")

# Communication libraries
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: MQTT not available, mobile app integration disabled")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: WebSockets not available, mobile app integration limited")

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class DroneState(Enum):
    """Drone operational states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ARMED = "armed"
    TAKEOFF = "takeoff"
    NAVIGATION = "navigation"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    PAYLOAD_DELIVERY = "payload_delivery"
    WATER_LANDING = "water_landing"
    WATER_SURFACE = "water_surface"
    EMERGENCY = "emergency"
    LANDING = "landing"
    DISARMED = "disarmed"

class MissionType(Enum):
    """Mission types for disaster response"""
    SEARCH_AND_RESCUE = "search_rescue"
    MEDICAL_DELIVERY = "medical_delivery"
    COMMUNICATION_RELAY = "communication_relay"
    DAMAGE_ASSESSMENT = "damage_assessment"

@dataclass
class DroneConfig:
    """Configuration parameters for the drone system"""
    # Safety parameters
    MIN_ALTITUDE: float = 2.0
    MAX_ALTITUDE: float = 120.0
    SAFE_DISTANCE: float = 3.0
    EMERGENCY_STOP_DISTANCE: float = 1.5
    
    # Navigation parameters
    MAX_VELOCITY: float = 8.0
    MAX_ANGULAR_VELOCITY: float = 1.5
    WAYPOINT_TOLERANCE: float = 2.0
    HEADING_TOLERANCE: float = 0.2
    
    # Sensor parameters
    CAMERA_FOV: float = 69.4
    LIDAR_RANGE: float = 100.0
    UPDATE_RATE: float = 20.0
    LIDAR_UPDATE_RATE: float = 10.0
    
    # Mission parameters
    PAYLOAD_CAPACITY: float = 5.0
    WINCH_MAX_LENGTH: float = 50.0
    WATER_DETECTION_THRESHOLD: float = 1.0
    
    # Communication
    MAVLINK_CONNECTION: str = "udp:127.0.0.1:14550"
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    WEBSOCKET_PORT: int = 8765
    
    # Mapping parameters
    MAP_RESOLUTION: float = 0.2
    MAP_SIZE_X: int = 2000
    MAP_SIZE_Y: int = 2000
    
    # AI parameters
    YOLO_CONFIDENCE_THRESHOLD: float = 0.4
    DETECTION_NMS_THRESHOLD: float = 0.5

CONFIG = DroneConfig()

# ============================================================================
# MESSAGE DEFINITIONS (Replacing ROS2 messages)
# ============================================================================

@dataclass
class Header:
    """Message header with timestamp and frame ID"""
    timestamp: float = field(default_factory=time.time)
    frame_id: str = "base_link"

@dataclass
class Point:
    """3D point"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Quaternion:
    """Quaternion representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

@dataclass
class Vector3:
    """3D vector"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Twist:
    """Linear and angular velocity"""
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)

@dataclass
class PoseStamped:
    """Stamped pose message"""
    header: Header = field(default_factory=Header)
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)

@dataclass
class NavSatFix:
    """GPS fix message"""
    header: Header = field(default_factory=Header)
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    position_covariance: List[float] = field(default_factory=lambda: [0.0]*9)

@dataclass
class Imu:
    """IMU data message"""
    header: Header = field(default_factory=Header)
    orientation: Quaternion = field(default_factory=Quaternion)
    linear_acceleration: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)

@dataclass
class BatteryState:
    """Battery status message"""
    header: Header = field(default_factory=Header)
    voltage: float = 0.0
    current: float = 0.0
    charge: float = 0.0
    capacity: float = 0.0
    percentage: float = 0.0

@dataclass
class Image:
    """Image message"""
    header: Header = field(default_factory=Header)
    width: int = 0
    height: int = 0
    encoding: str = "bgr8"
    data: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class PointCloud2:
    """Point cloud message"""
    header: Header = field(default_factory=Header)
    width: int = 0
    height: int = 0
    points: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class OccupancyGrid:
    """Occupancy grid map message"""
    header: Header = field(default_factory=Header)
    resolution: float = 0.05
    width: int = 0
    height: int = 0
    origin_x: float = 0.0
    origin_y: float = 0.0
    data: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class Path:
    """Navigation path message"""
    header: Header = field(default_factory=Header)
    poses: List[PoseStamped] = field(default_factory=list)

@dataclass
class Detection:
    """Object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[float, float]
    distance: float = float('inf')
    world_position: Optional[Tuple[float, float, float]] = None
    timestamp: float = 0.0

@dataclass
class Waypoint:
    """Waypoint definition for navigation"""
    x: float
    y: float
    z: float
    yaw: float = 0.0
    tolerance: float = CONFIG.WAYPOINT_TOLERANCE
    mission_type: MissionType = MissionType.SEARCH_AND_RESCUE
    action: str = "navigate"  # navigate, deliver, land, hover

@dataclass
class ObstacleInfo:
    """Obstacle information for path planning"""
    position: np.ndarray
    size: float
    height: float
    confidence: float = 1.0
    type: str = "unknown"
    timestamp: float = 0.0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quaternion_to_euler(quat: Quaternion) -> Tuple[float, float, float]:
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    return r.as_euler('xyz', degrees=False)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert euler angles to quaternion"""
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    q = r.as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate 3D distance between two points"""
    return np.sqrt(np.sum((p1 - p2)**2))

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# ============================================================================
# COMMUNICATION SYSTEM (Replacing ROS2 middleware)
# ============================================================================

class MessageBus:
    """Central message bus for inter-thread communication"""
    
    def __init__(self):
        self.topics = {}
        self.subscribers = {}
        self.publishers = {}
        self.services = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + ".MessageBus")
        
    def create_topic(self, topic_name: str, message_type: type, queue_size: int = 10):
        """Create a new topic"""
        with self.lock:
            if topic_name not in self.topics:
                self.topics[topic_name] = {
                    'queue': queue.Queue(maxsize=queue_size),
                    'message_type': message_type,
                    'subscribers': [],
                    'latest_message': None
                }
                self.logger.debug(f"Created topic: {topic_name}")
    
    def subscribe(self, topic_name: str, callback: Callable, queue_size: int = 10):
        """Subscribe to a topic"""
        with self.lock:
            if topic_name not in self.topics:
                self.create_topic(topic_name, object, queue_size)
            
            subscriber_id = f"{topic_name}_{len(self.subscribers)}"
            subscriber_queue = queue.Queue(maxsize=queue_size)
            
            self.subscribers[subscriber_id] = {
                'topic': topic_name,
                'callback': callback,
                'queue': subscriber_queue,
                'active': True
            }
            
            self.topics[topic_name]['subscribers'].append(subscriber_id)
            self.logger.debug(f"Created subscriber {subscriber_id} for topic {topic_name}")
            
            # Start callback thread
            callback_thread = threading.Thread(
                target=self._callback_worker,
                args=(subscriber_id,),
                daemon=True
            )
            callback_thread.start()
            
            return subscriber_id
    
    def publish(self, topic_name: str, message: Any):
        """Publish a message to a topic"""
        with self.lock:
            if topic_name not in self.topics:
                self.logger.warning(f"Topic {topic_name} does not exist")
                return
            
            # Update latest message
            self.topics[topic_name]['latest_message'] = message
            
            # Send to all subscribers
            for subscriber_id in self.topics[topic_name]['subscribers']:
                if subscriber_id in self.subscribers and self.subscribers[subscriber_id]['active']:
                    try:
                        self.subscribers[subscriber_id]['queue'].put_nowait(message)
                    except queue.Full:
                        self.logger.warning(f"Subscriber {subscriber_id} queue full, dropping message")
    
    def _callback_worker(self, subscriber_id: str):
        """Worker thread for handling subscriber callbacks"""
        while self.subscribers.get(subscriber_id, {}).get('active', False):
            try:
                subscriber = self.subscribers[subscriber_id]
                message = subscriber['queue'].get(timeout=1.0)
                subscriber['callback'](message)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in callback for {subscriber_id}: {e}")
    
    def create_service(self, service_name: str, service_callback: Callable):
        """Create a service"""
        with self.lock:
            self.services[service_name] = service_callback
            self.logger.debug(f"Created service: {service_name}")
    
    def call_service(self, service_name: str, request: Any) -> Any:
        """Call a service"""
        with self.lock:
            if service_name not in self.services:
                raise Exception(f"Service {service_name} not found")
            
            return self.services[service_name](request)
    
    def get_latest_message(self, topic_name: str) -> Optional[Any]:
        """Get the latest message from a topic"""
        with self.lock:
            topic_data = self.topics.get(topic_name)
            return topic_data['latest_message'] if topic_data else None
    
    def shutdown(self):
        """Shutdown the message bus"""
        with self.lock:
            # Deactivate all subscribers
            for subscriber_id in self.subscribers:
                self.subscribers[subscriber_id]['active'] = False
            
            self.logger.info("Message bus shutdown complete")

# Global message bus instance
message_bus = MessageBus()

class BaseNode:
    """Base class for all nodes (replacing ROS2 Node)"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.active = True
        self.message_bus = message_bus
        self.subscriptions = []
        self.publishers = {}
        self.services = {}
        self.timers = []
        self.threads = []
        
        self.logger.info(f"Node {name} initialized")
    
    def create_subscription(self, topic_name: str, message_type: type, callback: Callable, queue_size: int = 10):
        """Create a subscription"""
        subscriber_id = self.message_bus.subscribe(topic_name, callback, queue_size)
        self.subscriptions.append(subscriber_id)
        self.logger.debug(f"Created subscription to {topic_name}")
        return subscriber_id
    
    def create_publisher(self, topic_name: str, message_type: type, queue_size: int = 10):
        """Create a publisher"""
        self.message_bus.create_topic(topic_name, message_type, queue_size)
        
        def publish_func(message):
            self.message_bus.publish(topic_name, message)
        
        self.publishers[topic_name] = publish_func
        self.logger.debug(f"Created publisher for {topic_name}")
        return publish_func
    
    def create_service(self, service_name: str, service_callback: Callable):
        """Create a service"""
        self.message_bus.create_service(service_name, service_callback)
        self.services[service_name] = service_callback
        self.logger.debug(f"Created service {service_name}")
    
    def call_service(self, service_name: str, request: Any) -> Any:
        """Call a service"""
        return self.message_bus.call_service(service_name, request)
    
    def create_timer(self, timer_period: float, callback: Callable):
        """Create a timer"""
        def timer_worker():
            while self.active:
                try:
                    callback()
                    time.sleep(timer_period)
                except Exception as e:
                    self.logger.error(f"Timer callback error: {e}")
        
        timer_thread = threading.Thread(target=timer_worker, daemon=True)
        timer_thread.start()
        self.timers.append(timer_thread)
        self.logger.debug(f"Created timer with period {timer_period}s")
        return timer_thread
    
    def start_thread(self, target: Callable, args: tuple = (), kwargs: dict = {}):
        """Start a new thread"""
        thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        self.threads.append(thread)
        return thread
    
    def shutdown(self):
        """Shutdown the node"""
        self.active = False
        self.logger.info(f"Node {self.name} shutting down")

# ============================================================================
# MAVLINK INTERFACE (Replacing MAVROS)
# ============================================================================

class MAVLinkInterface(BaseNode):
    """MAVLink communication interface"""
    
    def __init__(self, connection_string: str = CONFIG.MAVLINK_CONNECTION):
        super().__init__("mavlink_interface")
        self.connection_string = connection_string
        self.connection = None
        self.heartbeat_thread = None
        self.message_thread = None
        
        # Publishers
        self.state_pub = self.create_publisher("mavlink/state", dict)
        self.pose_pub = self.create_publisher("mavlink/pose", PoseStamped)
        self.velocity_pub = self.create_publisher("mavlink/velocity", Twist)
        self.gps_pub = self.create_publisher("mavlink/gps", NavSatFix)
        self.imu_pub = self.create_publisher("mavlink/imu", Imu)
        self.battery_pub = self.create_publisher("mavlink/battery", BatteryState)
        
        # Subscribers
        self.create_subscription("control/velocity", Twist, self.velocity_callback)
        self.create_subscription("control/position", PoseStamped, self.position_callback)
        
        # Services
        self.create_service("mavlink/arm", self.arm_service)
        self.create_service("mavlink/takeoff", self.takeoff_service)
        self.create_service("mavlink/land", self.land_service)
        self.create_service("mavlink/set_mode", self.set_mode_service)
        
        # State
        self.armed = False
        self.mode = "STABILIZE"
        self.connected = False
        
        # Start connection
        self.start_thread(self.connect)
    
    def connect(self):
        """Connect to MAVLink autopilot"""
        if not PYMAVLINK_AVAILABLE:
            self.logger.warning("PyMAVLink not available, using mock connection")
            self.connected = True
            self.start_thread(self.mock_message_loop)
            return
        
        try:
            self.logger.info(f"Connecting to MAVLink at {self.connection_string}")
            self.connection = mavutil.mavlink_connection(self.connection_string, timeout=10)
            
            # Wait for heartbeat
            self.logger.info("Waiting for heartbeat...")
            self.connection.wait_heartbeat()
            self.logger.info("Heartbeat received, connection established")
            
            self.connected = True
            
            # Start message loops
            self.start_thread(self.heartbeat_loop)
            self.start_thread(self.message_loop)
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MAVLink: {e}")
            self.connected = False
    
    def heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.active and self.connected:
            try:
                if self.connection:
                    self.connection.mav.heartbeat_send(
                        mavlink.MAV_TYPE_GCS,
                        mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                time.sleep(1.0)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
    
    def message_loop(self):
        """Process incoming MAVLink messages"""
        while self.active and self.connected:
            try:
                if not self.connection:
                    time.sleep(0.1)
                    continue
                
                msg = self.connection.recv_match(blocking=False, timeout=0.1)
                if msg:
                    self.process_message(msg)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Message loop error: {e}")
                time.sleep(0.1)
    
    def mock_message_loop(self):
        """Mock message loop for testing without hardware"""
        altitude = 0.0
        lat = 37.7749  # San Francisco
        lon = -122.4194
        
        while self.active:
            try:
                # Mock GPS
                gps_msg = NavSatFix()
                gps_msg.latitude = lat + random.uniform(-0.0001, 0.0001)
                gps_msg.longitude = lon + random.uniform(-0.0001, 0.0001)
                gps_msg.altitude = altitude + random.uniform(-0.1, 0.1)
                self.gps_pub(gps_msg)
                
                # Mock IMU
                imu_msg = Imu()
                imu_msg.orientation = Quaternion(w=1.0)
                imu_msg.linear_acceleration = Vector3(x=0.0, y=0.0, z=9.81)
                imu_msg.angular_velocity = Vector3()
                self.imu_pub(imu_msg)
                
                # Mock battery
                battery_msg = BatteryState()
                battery_msg.voltage = 12.5 + random.uniform(-0.2, 0.2)
                battery_msg.percentage = max(0, 85.0 + random.uniform(-5, 2))
                self.battery_pub(battery_msg)
                
                # Mock state
                state = {
                    'armed': self.armed,
                    'mode': self.mode,
                    'connected': True
                }
                self.state_pub(state)
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Mock message loop error: {e}")
    
    def process_message(self, msg):
        """Process a MAVLink message"""
        try:
            if msg.get_type() == 'HEARTBEAT':
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                
                # Publish state
                state = {
                    'armed': self.armed,
                    'mode': self.mode,
                    'connected': True
                }
                self.state_pub(state)
            
            elif msg.get_type() == 'GLOBAL_POSITION_INT':
                # Publish GPS
                gps_msg = NavSatFix()
                gps_msg.latitude = msg.lat / 1e7
                gps_msg.longitude = msg.lon / 1e7
                gps_msg.altitude = msg.alt / 1000.0
                self.gps_pub(gps_msg)
            
            elif msg.get_type() == 'ATTITUDE':
                # Publish IMU orientation
                imu_msg = Imu()
                imu_msg.orientation = euler_to_quaternion(msg.roll, msg.pitch, msg.yaw)
                self.imu_pub(imu_msg)
            
            elif msg.get_type() == 'BATTERY_STATUS':
                # Publish battery state
                battery_msg = BatteryState()
                battery_msg.voltage = msg.voltages[0] / 1000.0 if msg.voltages else 0.0
                battery_msg.current = msg.current_battery / 100.0 if msg.current_battery != -1 else 0.0
                battery_msg.percentage = msg.battery_remaining if msg.battery_remaining != -1 else 0.0
                self.battery_pub(battery_msg)
                
        except Exception as e:
            self.logger.error(f"Error processing message {msg.get_type()}: {e}")
    
    def velocity_callback(self, msg: Twist):
        """Handle velocity command"""
        if not self.connected or not self.connection:
            return
        
        try:
            # Send velocity setpoint
            self.connection.mav.set_position_target_local_ned_send(
                0, 0, 0,
                mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,  # Velocity control
                0, 0, 0,  # Position (ignored)
                msg.linear.x, msg.linear.y, msg.linear.z,  # Velocity
                0, 0, 0,  # Acceleration (ignored)
                0, 0  # Yaw, yaw_rate (ignored)
            )
        except Exception as e:
            self.logger.error(f"Error sending velocity command: {e}")
    
    def position_callback(self, msg: PoseStamped):
        """Handle position command"""
        if not self.connected or not self.connection:
            return
        
        try:
            # Convert quaternion to yaw
            _, _, yaw = quaternion_to_euler(msg.orientation)
            
            # Send position setpoint
            self.connection.mav.set_position_target_local_ned_send(
                0, 0, 0,
                mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,  # Position control
                msg.position.x, msg.position.y, -msg.position.z,  # Position (NED)
                0, 0, 0,  # Velocity (ignored)
                0, 0, 0,  # Acceleration (ignored)
                yaw, 0  # Yaw, yaw_rate
            )
        except Exception as e:
            self.logger.error(f"Error sending position command: {e}")
    
    def arm_service(self, request):
        """Arm/disarm the vehicle"""
        try:
            arm = bool(request.get('arm', True))
            
            if not self.connection:
                return {'success': False, 'message': 'Not connected'}
            
            self.connection.mav.command_long_send(
                0, 0,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1 if arm else 0, 0, 0, 0, 0, 0, 0
            )
            
            return {'success': True, 'message': f'{"Armed" if arm else "Disarmed"} command sent'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def takeoff_service(self, request):
        """Takeoff to specified altitude"""
        try:
            altitude = float(request.get('altitude', 10.0))
            
            if not self.connection:
                return {'success': False, 'message': 'Not connected'}
            
            self.connection.mav.command_long_send(
                0, 0,
                mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0, 0, 0, altitude
            )
            
            return {'success': True, 'message': f'Takeoff to {altitude}m command sent'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def land_service(self, request):
        """Land the vehicle"""
        try:
            if not self.connection:
                return {'success': False, 'message': 'Not connected'}
            
            self.connection.mav.command_long_send(
                0, 0,
                mavlink.MAV_CMD_NAV_LAND,
                0,
                0, 0, 0, 0, 0, 0, 0
            )
            
            return {'success': True, 'message': 'Land command sent'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def set_mode_service(self, request):
        """Set flight mode"""
        try:
            mode_name = request.get('mode', 'GUIDED')
            
            if not self.connection:
                return {'success': False, 'message': 'Not connected'}
            
            # Get mode mapping
            mode_map = self.connection.mode_mapping()
            if mode_name not in mode_map:
                return {'success': False, 'message': f'Unknown mode: {mode_name}'}
            
            mode_id = mode_map[mode_name]
            
            self.connection.mav.set_mode_send(
                0,
                mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            
            self.mode = mode_name
            return {'success': True, 'message': f'Mode set to {mode_name}'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

# ============================================================================
# PERCEPTION LAYER
# ============================================================================

class YOLOPerception(BaseNode):
    """YOLOv8-based object detection for disaster response scenarios"""
    
    def __init__(self):
        super().__init__("yolo_perception")
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extended disaster response classes
        self.disaster_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light',
            14: 'bird', 15: 'cat', 16: 'dog', 24: 'backpack', 39: 'bottle',
            56: 'chair', 57: 'couch', 58: 'potted_plant', 62: 'tv', 63: 'laptop',
            # Custom disaster classes
            80: 'debris', 81: 'fire', 82: 'flood_water', 83: 'building_damage',
            84: 'medical_supplies', 85: 'emergency_vehicle', 86: 'survivor_signal'
        }
        
        self.priority_classes = ['person', 'medical_supplies', 'emergency_vehicle',
                                'fire', 'debris', 'building_damage']
        
        # Publishers
        self.detection_pub = self.create_publisher("perception/detections", list)
        self.priority_detection_pub = self.create_publisher("perception/priority_detections", list)
        
        # Subscribers
        self.create_subscription("sensors/camera/image", Image, self.image_callback)
        
        # Initialize YOLO
        self._init_yolo()
        
        self.last_detections = []
        self.detection_history = deque(maxlen=10)
    
    def _init_yolo(self):
        """Initialize YOLOv8 model"""
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
                self.logger.info(f"YOLOv8 initialized on {self.device}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLOv8: {e}")
                self.model = None
        else:
            self.logger.warning("YOLOv8 not available, using mock detection")
    
    def image_callback(self, msg: Image):
        """Process incoming images"""
        try:
            detections = self.detect_objects(msg.data)
            self.last_detections = detections
            self.detection_history.append(detections)
            
            # Publish all detections
            self.detection_pub(detections)
            
            # Publish priority detections
            priority_detections = self.get_priority_detections()
            self.priority_detection_pub(priority_detections)
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
    
    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image and return detections"""
        if self.model is None:
            return self._mock_detection(image)
        
        try:
            results = self.model(image, 
                               conf=CONFIG.YOLO_CONFIDENCE_THRESHOLD,
                               iou=CONFIG.DETECTION_NMS_THRESHOLD, 
                               verbose=False)
            
            detections = []
            current_time = time.time()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.disaster_classes.get(class_id, 'unknown')
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=[int(x1), int(y1), int(x2), int(y2)],
                            center=((x1 + x2) / 2, (y1 + y2) / 2),
                            timestamp=current_time
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return self.last_detections
    
    def _mock_detection(self, image: np.ndarray) -> List[Detection]:
        """Mock detection for testing"""
        if image is None or image.size == 0:
            return []
            
        h, w = image.shape[:2] if len(image.shape) >= 2 else (480, 640)
        current_time = time.time()
        mock_detections = []
        
        # Simulate person detection
        if np.random.random() < 0.3:
            mock_detections.append(Detection(
                class_id=0, class_name='person', confidence=0.85,
                bbox=[w//4, h//4, w//2, h//2],
                center=(w//3, h//3), timestamp=current_time
            ))
        
        # Simulate vehicle detection
        if np.random.random() < 0.2:
            mock_detections.append(Detection(
                class_id=2, class_name='car', confidence=0.72,
                bbox=[w//8, h//2, w//3, 3*h//4],
                center=(w//5, 5*h//8), timestamp=current_time
            ))
        
        return mock_detections
    
    def get_priority_detections(self) -> List[Detection]:
        """Get detections of priority objects"""
        priority_detections = []
        for detection in self.last_detections:
            if detection.class_name in self.priority_classes:
                priority_detections.append(detection)
        
        priority_detections.sort(key=lambda x: x.confidence, reverse=True)
        return priority_detections

class RealSenseProcessor(BaseNode):
    """Intel RealSense camera processing for RGB-D data"""
    
    def __init__(self):
        super().__init__("realsense_processor")
        self.pipeline = None
        self.config = None
        self.intrinsics = None
        self.depth_scale = 0.001
        
        # Publishers
        self.image_pub = self.create_publisher("sensors/camera/image", Image)
        self.depth_pub = self.create_publisher("sensors/camera/depth", Image)
        
        # Camera intrinsics for D435 (approximate)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Initialize RealSense
        self._init_realsense()
        
        # Start capture thread
        self.start_thread(self.capture_loop)
    
    def _init_realsense(self):
        """Initialize RealSense camera"""
        if REALSENSE_AVAILABLE:
            try:
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                profile = self.pipeline.start(self.config)
                
                depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
                self.intrinsics = depth_profile.get_intrinsics()
                
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                
                self.logger.info("RealSense camera initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize RealSense: {e}")
                self.pipeline = None
        else:
            self.logger.warning("RealSense not available, using mock data")
    
    def capture_loop(self):
        """Main capture loop"""
        while self.active:
            try:
                color_image, depth_image = self.get_frames()
                
                if color_image is not None:
                    # Publish color image
                    color_msg = Image()
                    color_msg.width = color_image.shape[1]
                    color_msg.height = color_image.shape[0]
                    color_msg.encoding = "bgr8"
                    color_msg.data = color_image
                    self.image_pub(color_msg)
                
                if depth_image is not None:
                    # Publish depth image
                    depth_msg = Image()
                    depth_msg.width = depth_image.shape[1]
                    depth_msg.height = depth_image.shape[0]
                    depth_msg.encoding = "16UC1"
                    depth_msg.data = depth_image
                    self.depth_pub(depth_msg)
                
                time.sleep(1.0 / 30.0)  # 30 FPS
                
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get RGB and depth frames"""
        if self.pipeline is None:
            return self._mock_frames()
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return self._mock_frames()
            
            # Apply filters
            depth_frame = self._apply_depth_filters(depth_frame)
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            self.logger.error(f"RealSense frame capture error: {e}")
            return self._mock_frames()
    
    def _apply_depth_filters(self, depth_frame):
        """Apply depth filters for better quality"""
        decimation = rs.decimation_filter()
        depth_frame = decimation.process(depth_frame)
        
        spatial = rs.spatial_filter()
        depth_frame = spatial.process(depth_frame)
        
        temporal = rs.temporal_filter()
        depth_frame = temporal.process(depth_frame)
        
        return depth_frame
    
    def _mock_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock RGB-D frames"""
        color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(color_image, (200, 150), (400, 350), (0, 255, 0), -1)
        cv2.rectangle(color_image, (450, 200), (600, 400), (0, 0, 255), -1)
        cv2.circle(color_image, (100, 100), 50, (255, 0, 0), -1)
        
        depth_image = np.random.randint(1000, 4000, (480, 640), dtype=np.uint16)
        depth_image[150:350, 200:400] = 2000
        depth_image[200:400, 450:600] = 3500
        depth_image[50:150, 50:150] = 1500
        
        return color_image, depth_image
    
    def estimate_distance_to_bbox(self, depth_image: np.ndarray, bbox: List[int]) -> float:
        """Estimate distance to object using bounding box in depth image"""
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        x1 = max(0, min(x1, depth_image.shape[1] - 1))
        x2 = max(0, min(x2, depth_image.shape[1] - 1))
        y1 = max(0, min(y1, depth_image.shape[0] - 1))
        y2 = max(0, min(y2, depth_image.shape[0] - 1))
        
        if x1 >= x2 or y1 >= y2:
            return float('inf')
        
        # Extract region of interest from depth image
        roi = depth_image[y1:y2, x1:x2]
        
        # Convert to meters
        roi_meters = roi * self.depth_scale
        
        # Filter out invalid depths
        valid_depths = roi_meters[(roi_meters > 0.1) & (roi_meters < 50.0)]
        
        if len(valid_depths) > 10:
            return float(np.percentile(valid_depths, 25))
        else:
            return float('inf')

class LiDARProcessor(BaseNode):
    """LiDAR point cloud processing for obstacle mapping and navigation"""
    
    def __init__(self):
        super().__init__("lidar_processor")
        self.range_limit = CONFIG.LIDAR_RANGE
        self.angle_resolution = 1.0  # degrees
        self.height_filter_min = -2.0
        self.height_filter_max = 10.0
        self.cluster_tolerance = 1.5
        self.min_cluster_size = 5
        
        # Publishers
        self.pointcloud_pub = self.create_publisher("sensors/lidar/pointcloud", PointCloud2)
        self.obstacles_pub = self.create_publisher("sensors/lidar/obstacles", list)
        
        # Store recent point clouds
        self.recent_clouds = deque(maxlen=5)
        self.obstacle_tracker = {}
        
        # Start mock LiDAR thread
        self.start_thread(self.mock_lidar_loop)
    
    def mock_lidar_loop(self):
        """Mock LiDAR data generation for testing"""
        while self.active:
            try:
                # Generate mock point cloud
                mock_cloud = self._generate_mock_pointcloud()
                
                # Process point cloud
                lidar_data = self.process_pointcloud(mock_cloud)
                
                # Publish point cloud
                pc_msg = PointCloud2()
                pc_msg.points = mock_cloud
                self.pointcloud_pub(pc_msg)
                
                # Publish obstacles
                if 'obstacles' in lidar_data:
                    self.obstacles_pub(lidar_data['obstacles'])
                
                time.sleep(1.0 / CONFIG.LIDAR_UPDATE_RATE)
                
            except Exception as e:
                self.logger.error(f"Mock LiDAR error: {e}")
                time.sleep(0.1)
    
    def _generate_mock_pointcloud(self) -> np.ndarray:
        """Generate mock point cloud data"""
        points = []
        
        # Generate 360-degree scan
        for angle in np.linspace(0, 2*np.pi, 360):
            # Base range with some obstacles
            base_range = 20.0 + 10.0 * np.sin(angle * 3)
            
            # Add some obstacles
            if 0.5 < angle < 1.0 or 3.0 < angle < 3.5:
                range_val = 5.0 + 2.0 * np.random.random()
            else:
                range_val = base_range + 2.0 * np.random.random()
            
            # Convert to Cartesian
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)
            z = np.random.uniform(-0.5, 2.0)  # Height variation
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def process_pointcloud(self, pointcloud: np.ndarray) -> Dict:
        """Process LiDAR point cloud to extract obstacle information"""
        try:
            if pointcloud.size == 0:
                return {'obstacles': [], 'histogram': []}
            
            # Store in recent clouds
            self.recent_clouds.append(pointcloud)
            
            # Filter by range and height
            distances_2d = np.linalg.norm(pointcloud[:, :2], axis=1)
            valid_indices = (
                (distances_2d < self.range_limit) &
                (pointcloud[:, 2] > self.height_filter_min) &
                (pointcloud[:, 2] < self.height_filter_max)
            )
            
            filtered_points = pointcloud[valid_indices]
            
            if len(filtered_points) == 0:
                return {'obstacles': [], 'histogram': []}
            
            # Convert to polar coordinates
            angles = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])
            distances = np.linalg.norm(filtered_points[:, :2], axis=1)
            
            # Create histogram
            histogram = self._create_polar_histogram(angles, distances)
            
            # Detect obstacles
            obstacles = self._detect_obstacles_clustering(filtered_points)
            
            return {
                'obstacles': obstacles,
                'histogram': histogram,
                'raw_points': filtered_points,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"LiDAR processing error: {e}")
            return {'obstacles': [], 'histogram': []}
    
    def _create_polar_histogram(self, angles: np.ndarray, distances: np.ndarray) -> List[float]:
        """Create polar histogram for VFH algorithm"""
        bins = int(360 / self.angle_resolution)
        histogram = np.zeros(bins)
        
        for angle, distance in zip(angles, distances):
            angle_deg = np.degrees(angle)
            bin_idx = int((angle_deg + 180) / self.angle_resolution) % bins
            
            if distance > 0.1:
                weight = 1.0 / (distance**2 + 0.1)
                histogram[bin_idx] += weight
        
        # Apply smoothing
        kernel = np.array([0.25, 0.5, 0.25])
        histogram_smooth = np.convolve(
            np.concatenate([histogram[-1:], histogram, histogram[:1]]),
            kernel, mode='valid'
        )
        
        return histogram_smooth.tolist()
    
    def _detect_obstacles_clustering(self, points: np.ndarray) -> List[Dict]:
        """Detect discrete obstacles using DBSCAN clustering"""
        if len(points) < self.min_cluster_size:
            return []
        
        try:
            clustering = DBSCAN(eps=self.cluster_tolerance,
                              min_samples=self.min_cluster_size).fit(points[:, :2])
            
            obstacles = []
            current_time = time.time()
            
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:  # Noise points
                    continue
                
                cluster_points = points[clustering.labels_ == cluster_id]
                centroid = np.mean(cluster_points, axis=0)
                
                distances_from_centroid = np.linalg.norm(
                    cluster_points - centroid, axis=1)
                cluster_size = np.max(distances_from_centroid)
                
                height_min = np.min(cluster_points[:, 2])
                height_max = np.max(cluster_points[:, 2])
                
                confidence = min(1.0, len(cluster_points) / 50.0)
                obstacle_type = self._classify_obstacle(cluster_size, height_max - height_min)
                
                obstacle = {
                    'id': f"obs_{cluster_id}_{int(current_time)}",
                    'position': centroid.tolist(),
                    'size': float(cluster_size),
                    'height_min': float(height_min),
                    'height_max': float(height_max),
                    'points_count': len(cluster_points),
                    'confidence': confidence,
                    'type': obstacle_type,
                    'timestamp': current_time
                }
                obstacles.append(obstacle)
            
            obstacles.sort(key=lambda x: np.linalg.norm(x['position'][:2]))
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Obstacle clustering error: {e}")
            return []
    
    def _classify_obstacle(self, size: float, height: float) -> str:
        """Classify obstacle type based on size and height"""
        if height > 3.0:
            return "building" if size > 2.0 else "tree"
        elif height > 1.5:
            return "vehicle" if size > 1.0 else "pole"
        else:
            return "debris" if size > 0.5 else "small_object"

# ============================================================================
# SLAM AND NAVIGATION
# ============================================================================

class ExtendedKalmanFilter:
    """Extended Kalman Filter for drone state estimation"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.state_dim = 12
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1000
        
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.1,  # Position
                         0.5, 0.5, 0.5,  # Velocity
                         0.01, 0.01, 0.02,  # Orientation
                         0.1, 0.1, 0.1])  # Angular velocity
        
        # Measurement noise matrices
        self.R_gps = np.diag([2.0, 2.0, 5.0])
        self.R_imu_accel = np.diag([0.5, 0.5, 0.5])
        
        self.last_update_time = time.time()
    
    def predict(self, dt: float, imu_data: Optional[Dict] = None):
        """Prediction step using motion model"""
        if dt <= 0 or dt > 1.0:
            dt = 0.05
        
        # State transition matrix
        F = np.eye(self.state_dim)
        F[0, 3] = dt  # x = x + vx*dt
        F[1, 4] = dt  # y = y + vy*dt
        F[2, 5] = dt  # z = z + vz*dt
        F[6, 9] = dt  # roll = roll + wx*dt
        F[7, 10] = dt  # pitch = pitch + wy*dt
        F[8, 11] = dt  # yaw = yaw + wz*dt
        
        # Predict state
        self.state = F @ self.state
        
        # Add IMU measurements
        if imu_data:
            accel = imu_data.get('linear_acceleration', [0, 0, 0])
            gyro = imu_data.get('angular_velocity', [0, 0, 0])
            
            self.state[3:6] += np.array(accel) * dt
            self.state[9:12] = np.array(gyro)
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # Normalize yaw
        self.state[8] = normalize_angle(self.state[8])
    
    def update_gps(self, gps_measurement: np.ndarray):
        """Update with GPS position measurement"""
        if len(gps_measurement) != 3:
            return
        
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        
        self._update_step(gps_measurement, H, self.R_gps)
    
    def _update_step(self, measurement: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Generic EKF update step"""
        try:
            y = measurement - H @ self.state
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            self.state = self.state + K @ y
            self.P = (np.eye(self.state_dim) - K @ H) @ self.P
            
            self.state[8] = normalize_angle(self.state[8])
            
        except np.linalg.LinAlgError:
            if self.logger:
                self.logger.warning("Singular matrix in EKF update")
    
    def get_position(self) -> np.ndarray:
        return self.state[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        return self.state[3:6].copy()
    
    def get_orientation(self) -> np.ndarray:
        return self.state[6:9].copy()

class SLAMProcessor(BaseNode):
    """Main SLAM processor combining localization and mapping"""
    
    def __init__(self):
        super().__init__("slam_processor")
        self.ekf = ExtendedKalmanFilter(self.logger)
        
        # State tracking
        self.current_pose = np.zeros(3)
        self.current_orientation = np.zeros(3)
        self.current_velocity = np.zeros(3)
        
        # Publishers
        self.pose_pub = self.create_publisher("slam/pose", PoseStamped)
        self.velocity_pub = self.create_publisher("slam/velocity", Twist)
        
        # Subscribers
        self.create_subscription("mavlink/gps", NavSatFix, self.gps_callback)
        self.create_subscription("mavlink/imu", Imu, self.imu_callback)
        
        # Create update timer
        self.create_timer(0.05, self.update_slam)  # 20 Hz
        
        self.last_update_time = time.time()
    
    def gps_callback(self, msg: NavSatFix):
        """Handle GPS updates"""
        try:
            # Simple conversion for demo (use proper geodetic transforms in real system)
            x = msg.longitude * 111000 * np.cos(np.radians(msg.latitude))
            y = msg.latitude * 111000
            z = msg.altitude
            
            self.ekf.update_gps(np.array([x, y, z]))
        except Exception as e:
            self.logger.error(f"GPS callback error: {e}")
    
    def imu_callback(self, msg: Imu):
        """Handle IMU updates"""
        try:
            imu_data = {
                'linear_acceleration': [msg.linear_acceleration.x,
                                      msg.linear_acceleration.y,
                                      msg.linear_acceleration.z],
                'angular_velocity': [msg.angular_velocity.x,
                                   msg.angular_velocity.y,
                                   msg.angular_velocity.z]
            }
            
            # Store for prediction step
            self.last_imu_data = imu_data
        except Exception as e:
            self.logger.error(f"IMU callback error: {e}")
    
    def update_slam(self):
        """Main SLAM update"""
        try:
            current_time = time.time()
            dt = current_time - self.last_update_time
            
            # EKF prediction
            imu_data = getattr(self, 'last_imu_data', None)
            self.ekf.predict(dt, imu_data)
            
            # Update state estimates
            self.current_pose = self.ekf.get_position()
            self.current_velocity = self.ekf.get_velocity()
            self.current_orientation = self.ekf.get_orientation()
            
            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.position = Point(x=self.current_pose[0],
                                    y=self.current_pose[1],
                                    z=self.current_pose[2])
            pose_msg.orientation = euler_to_quaternion(*self.current_orientation)
            self.pose_pub(pose_msg)
            
            # Publish velocity
            vel_msg = Twist()
            vel_msg.linear = Vector3(x=self.current_velocity[0],
                                   y=self.current_velocity[1],
                                   z=self.current_velocity[2])
            self.velocity_pub(vel_msg)
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"SLAM update error: {e}")
    
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current pose estimate"""
        return self.current_pose.copy(), self.current_orientation.copy()
    
    def get_current_velocity(self) -> np.ndarray:
        """Get current velocity estimate"""
        return self.current_velocity.copy()

# ============================================================================
# PATH PLANNING AND CONTROL
# ============================================================================

class PIDController:
    """PID controller for various control loops"""
    
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = None, integral_limit: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error: float, dt: Optional[float] = None) -> float:
        """Update PID controller with current error"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        if self.integral_limit:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt
        
        # Combine terms
        output = proportional + integral + derivative
        
        # Limit output
        if self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)
        
        # Update for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class NavigationController(BaseNode):
    """Navigation and path following controller"""
    
    def __init__(self):
        super().__init__("navigation_controller")
        
        # PID controllers for position control
        self.pid_x = PIDController(2.0, 0.1, 0.5, output_limit=CONFIG.MAX_VELOCITY)
        self.pid_y = PIDController(2.0, 0.1, 0.5, output_limit=CONFIG.MAX_VELOCITY)
        self.pid_z = PIDController(3.0, 0.2, 0.8, output_limit=CONFIG.MAX_VELOCITY*0.5)
        self.pid_yaw = PIDController(2.0, 0.0, 0.1, output_limit=CONFIG.MAX_ANGULAR_VELOCITY)
        
        # Current target
        self.target_position = np.zeros(3)
        self.target_yaw = 0.0
        self.position_control_active = False
        
        # Publishers
        self.velocity_cmd_pub = self.create_publisher("control/velocity", Twist)
        
        # Subscribers
        self.create_subscription("slam/pose", PoseStamped, self.pose_callback)
        self.create_subscription("control/goto", PoseStamped, self.goto_callback)
        self.create_subscription("control/velocity_direct", Twist, self.velocity_direct_callback)
        
        # Current state
        self.current_position = np.zeros(3)
        self.current_yaw = 0.0
        
        # Create control timer
        self.create_timer(0.05, self.control_loop)  # 20 Hz
    
    def pose_callback(self, msg: PoseStamped):
        """Handle pose updates from SLAM"""
        try:
            self.current_position = np.array([msg.position.x,
                                            msg.position.y,
                                            msg.position.z])
            
            # Extract yaw from quaternion
            _, _, self.current_yaw = quaternion_to_euler(msg.orientation)
            
        except Exception as e:
            self.logger.error(f"Pose callback error: {e}")
    
    def goto_callback(self, msg: PoseStamped):
        """Handle goto commands"""
        try:
            self.target_position = np.array([msg.position.x,
                                           msg.position.y,
                                           msg.position.z])
            
            _, _, self.target_yaw = quaternion_to_euler(msg.orientation)
            self.position_control_active = True
            
            self.logger.info(f"New target: {self.target_position}, yaw: {self.target_yaw:.2f}")
            
        except Exception as e:
            self.logger.error(f"Goto callback error: {e}")
    
    def velocity_direct_callback(self, msg: Twist):
        """Handle direct velocity commands (bypasses position control)"""
        try:
            self.position_control_active = False
            self.velocity_cmd_pub(msg)
        except Exception as e:
            self.logger.error(f"Direct velocity callback error: {e}")
    
    def control_loop(self):
        """Main control loop"""
        try:
            if not self.position_control_active:
                return
            
            # Calculate position errors
            position_error = self.target_position - self.current_position
            yaw_error = normalize_angle(self.target_yaw - self.current_yaw)
            
            # Check if target reached
            if np.linalg.norm(position_error) < CONFIG.WAYPOINT_TOLERANCE and \
               abs(yaw_error) < CONFIG.HEADING_TOLERANCE:
                # Target reached, stop
                self.position_control_active = False
                stop_cmd = Twist()
                self.velocity_cmd_pub(stop_cmd)
                self.logger.info("Target reached")
                return
            
            # Calculate velocity commands using PID controllers
            vel_x = self.pid_x.update(position_error[0])
            vel_y = self.pid_y.update(position_error[1])
            vel_z = self.pid_z.update(position_error[2])
            yaw_rate = self.pid_yaw.update(yaw_error)
            
            # Create and send velocity command
            cmd = Twist()
            cmd.linear = Vector3(x=vel_x, y=vel_y, z=vel_z)
            cmd.angular = Vector3(x=0.0, y=0.0, z=yaw_rate)
            
            self.velocity_cmd_pub(cmd)
            
        except Exception as e:
            self.logger.error(f"Control loop error: {e}")
    
    def fly_to_position(self, x: float, y: float, z: float, yaw: float = 0.0):
        """Command drone to fly to specific position"""
        target_msg = PoseStamped()
        target_msg.position = Point(x=x, y=y, z=z)
        target_msg.orientation = euler_to_quaternion(0, 0, yaw)
        self.goto_callback(target_msg)
    
    def stop(self):
        """Stop the drone"""
        self.position_control_active = False
        stop_cmd = Twist()
        self.velocity_cmd_pub(stop_cmd)

# ============================================================================
# PAYLOAD AND WINCH CONTROL
# ============================================================================

class PayloadController(BaseNode):
    """Payload delivery and winch control system"""
    
    def __init__(self):
        super().__init__("payload_controller")
        
        # Winch state
        self.winch_position = 0.0  # meters extended
        self.winch_speed = 2.0  # m/s
        self.payload_attached = False
        self.payload_weight = 0.0
        
        # Publishers
        self.winch_status_pub = self.create_publisher("payload/winch_status", dict)
        self.payload_status_pub = self.create_publisher("payload/payload_status", dict)
        
        # Services
        self.create_service("payload/extend_winch", self.extend_winch_service)
        self.create_service("payload/retract_winch", self.retract_winch_service)
        self.create_service("payload/release_payload", self.release_payload_service)
        self.create_service("payload/attach_payload", self.attach_payload_service)
        
        # Status timer
        self.create_timer(1.0, self.publish_status)
    
    def extend_winch_service(self, request):
        """Extend winch to specified length"""
        try:
            target_length = float(request.get('length', 10.0))
            
            if target_length > CONFIG.WINCH_MAX_LENGTH:
                return {'success': False, 'message': f'Length exceeds maximum: {CONFIG.WINCH_MAX_LENGTH}m'}
            
            if target_length < 0:
                return {'success': False, 'message': 'Length must be positive'}
            
            # Simulate winch extension
            self.start_thread(self._extend_winch, args=(target_length,))
            
            return {'success': True, 'message': f'Extending winch to {target_length}m'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def retract_winch_service(self, request):
        """Retract winch completely"""
        try:
            self.start_thread(self._retract_winch)
            return {'success': True, 'message': 'Retracting winch'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def release_payload_service(self, request):
        """Release attached payload"""
        try:
            if not self.payload_attached:
                return {'success': False, 'message': 'No payload attached'}
            
            # Simulate payload release
            self.payload_attached = False
            self.payload_weight = 0.0
            
            self.logger.info("Payload released")
            return {'success': True, 'message': 'Payload released'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def attach_payload_service(self, request):
        """Attach payload with specified weight"""
        try:
            weight = float(request.get('weight', 1.0))
            
            if weight > CONFIG.PAYLOAD_CAPACITY:
                return {'success': False, 'message': f'Payload too heavy: {weight}kg > {CONFIG.PAYLOAD_CAPACITY}kg'}
            
            self.payload_attached = True
            self.payload_weight = weight
            
            self.logger.info(f"Payload attached: {weight}kg")
            return {'success': True, 'message': f'Payload attached: {weight}kg'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _extend_winch(self, target_length: float):
        """Thread function to extend winch"""
        try:
            while self.winch_position < target_length and self.active:
                extension_step = min(self.winch_speed * 0.1, target_length - self.winch_position)
                self.winch_position += extension_step
                time.sleep(0.1)
            
            self.logger.info(f"Winch extended to {self.winch_position:.1f}m")
            
        except Exception as e:
            self.logger.error(f"Winch extension error: {e}")
    
    def _retract_winch(self):
        """Thread function to retract winch"""
        try:
            while self.winch_position > 0 and self.active:
                retraction_step = min(self.winch_speed * 0.1, self.winch_position)
                self.winch_position -= retraction_step
                time.sleep(0.1)
            
            self.winch_position = 0.0
            self.logger.info("Winch fully retracted")
            
        except Exception as e:
            self.logger.error(f"Winch retraction error: {e}")
    
    def publish_status(self):
        """Publish winch and payload status"""
        try:
            winch_status = {
                'position': self.winch_position,
                'max_length': CONFIG.WINCH_MAX_LENGTH,
                'speed': self.winch_speed,
                'timestamp': time.time()
            }
            self.winch_status_pub(winch_status)
            
            payload_status = {
                'attached': self.payload_attached,
                'weight': self.payload_weight,
                'capacity': CONFIG.PAYLOAD_CAPACITY,
                'timestamp': time.time()
            }
            self.payload_status_pub(payload_status)
            
        except Exception as e:
            self.logger.error(f"Status publish error: {e}")

# ============================================================================
# MISSION EXECUTOR
# ============================================================================

class MissionExecutor(BaseNode):
    """High-level mission execution and coordination"""
    
    def __init__(self):
        super().__init__("mission_executor")
        
        # Mission state
        self.current_mission = None
        self.mission_active = False
        self.mission_waypoints = []
        self.current_waypoint_index = 0
        
        # Subscribers for system state
        self.create_subscription("slam/pose", PoseStamped, self.pose_callback)
        self.create_subscription("perception/priority_detections", list, self.detection_callback)
        self.create_subscription("sensors/lidar/obstacles", list, self.obstacle_callback)
        
        # Publishers
        self.mission_status_pub = self.create_publisher("mission/status", dict)
        
        # Services
        self.create_service("mission/start_search_rescue", self.start_search_rescue_service)
        self.create_service("mission/start_delivery", self.start_delivery_service)
        self.create_service("mission/stop", self.stop_mission_service)
        
        # Current state
        self.current_position = np.zeros(3)
        self.priority_detections = []
        self.detected_obstacles = []
        
        # Create status timer
        self.create_timer(2.0, self.publish_mission_status)
    
    def pose_callback(self, msg: PoseStamped):
        """Handle pose updates"""
        self.current_position = np.array([msg.position.x, msg.position.y, msg.position.z])
    
    def detection_callback(self, detections: List[Detection]):
        """Handle priority detections"""
        self.priority_detections = detections
        
        if self.mission_active and detections:
            for detection in detections:
                if detection.class_name == 'person':
                    self.logger.critical(f"SURVIVOR DETECTED! Confidence: {detection.confidence:.2f}")
                    self.logger.critical(f"Position: {self.current_position}")
    
    def obstacle_callback(self, obstacles: List[Dict]):
        """Handle obstacle detections"""
        self.detected_obstacles = obstacles
    
    def start_search_rescue_service(self, request):
        """Start search and rescue mission"""
        try:
            waypoints = request.get('waypoints', [])
            altitude = float(request.get('altitude', 15.0))
            
            if not waypoints:
                return {'success': False, 'message': 'No waypoints provided'}
            
            # Convert waypoints
            self.mission_waypoints = []
            for wp in waypoints:
                waypoint = Waypoint(
                    x=float(wp['x']),
                    y=float(wp['y']),
                    z=altitude,
                    mission_type=MissionType.SEARCH_AND_RESCUE,
                    action='search'
                )
                self.mission_waypoints.append(waypoint)
            
            self.current_mission = MissionType.SEARCH_AND_RESCUE
            self.current_waypoint_index = 0
            self.mission_active = True
            
            # Start mission execution thread
            self.start_thread(self._execute_search_rescue_mission)
            
            self.logger.info(f"Started search and rescue mission with {len(waypoints)} waypoints")
            return {'success': True, 'message': f'Search and rescue mission started'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def start_delivery_service(self, request):
        """Start delivery mission"""
        try:
            target = request.get('target', {})
            payload_weight = float(request.get('payload_weight', 2.0))
            
            if not target:
                return {'success': False, 'message': 'No target provided'}
            
            # Create delivery waypoint
            delivery_waypoint = Waypoint(
                x=float(target['x']),
                y=float(target['y']),
                z=float(target.get('z', 20.0)),
                mission_type=MissionType.MEDICAL_DELIVERY,
                action='deliver'
            )
            
            self.mission_waypoints = [delivery_waypoint]
            self.current_mission = MissionType.MEDICAL_DELIVERY
            self.current_waypoint_index = 0
            self.mission_active = True
            
            # Attach payload
            payload_result = self.call_service("payload/attach_payload", {'weight': payload_weight})
            if not payload_result.get('success'):
                self.mission_active = False
                return {'success': False, 'message': f'Failed to attach payload: {payload_result.get("message")}'}
            
            # Start mission execution thread
            self.start_thread(self._execute_delivery_mission)
            
            self.logger.info(f"Started delivery mission to {target}")
            return {'success': True, 'message': 'Delivery mission started'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def stop_mission_service(self, request):
        """Stop current mission"""
        try:
            self.mission_active = False
            self.current_mission = None
            
            # Stop navigation
            self.call_service("navigation/stop", {})
            
            self.logger.info("Mission stopped")
            return {'success': True, 'message': 'Mission stopped'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _execute_search_rescue_mission(self):
        """Execute search and rescue mission"""
        try:
            self.logger.info("Executing search and rescue mission")
            
            for i, waypoint in enumerate(self.mission_waypoints):
                if not self.mission_active:
                    break
                
                self.current_waypoint_index = i
                self.logger.info(f"Flying to search waypoint {i+1}/{len(self.mission_waypoints)}")
                
                # Fly to waypoint
                self._fly_to_waypoint(waypoint)
                
                # Search pattern at waypoint
                self.logger.info("Performing search pattern...")
                self._perform_search_pattern(waypoint, radius=20.0)
                
                # Check for survivors
                time.sleep(5)  # Allow time for detection
            
            self.logger.info("Search and rescue mission completed")
            self.mission_active = False
            
        except Exception as e:
            self.logger.error(f"Search rescue mission error: {e}")
            self.mission_active = False
    
    def _execute_delivery_mission(self):
        """Execute delivery mission"""
        try:
            self.logger.info("Executing delivery mission")
            
            delivery_waypoint = self.mission_waypoints[0]
            
            # Fly to delivery point
            self.logger.info("Flying to delivery point")
            self._fly_to_waypoint(delivery_waypoint)
            
            # Lower payload
            self.logger.info("Lowering payload")
            winch_result = self.call_service("payload/extend_winch", {'length': 30.0})
            if winch_result.get('success'):
                time.sleep(15)  # Wait for winch extension
                
                # Release payload
                self.logger.info("Releasing payload")
                release_result = self.call_service("payload/release_payload", {})
                if release_result.get('success'):
                    time.sleep(2)
                    
                    # Retract winch
                    self.logger.info("Retracting winch")
                    self.call_service("payload/retract_winch", {})
                    time.sleep(15)  # Wait for retraction
            
            self.logger.info("Delivery mission completed")
            self.mission_active = False
            
        except Exception as e:
            self.logger.error(f"Delivery mission error: {e}")
            self.mission_active = False
    
    def _fly_to_waypoint(self, waypoint: Waypoint):
        """Fly to a specific waypoint"""
        try:
            # Use navigation controller to fly to waypoint
            nav_service_msg = {
                'x': waypoint.x,
                'y': waypoint.y,
                'z': waypoint.z,
                'yaw': waypoint.yaw
            }
            
            # Simplified waypoint following - in real system would integrate with navigation controller
            self.logger.info(f"Flying to waypoint: {waypoint.x:.1f}, {waypoint.y:.1f}, {waypoint.z:.1f}")
            
            # Wait for arrival (simplified)
            timeout = 60.0
            start_time = time.time()
            
            while self.mission_active and (time.time() - start_time) < timeout:
                distance = distance_3d(self.current_position, np.array([waypoint.x, waypoint.y, waypoint.z]))
                
                if distance < waypoint.tolerance:
                    self.logger.info(f"Reached waypoint (distance: {distance:.1f}m)")
                    return True
                
                time.sleep(0.5)
            
            self.logger.warning("Waypoint timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Fly to waypoint error: {e}")
            return False
    
    def _perform_search_pattern(self, center_waypoint: Waypoint, radius: float = 20.0):
        """Perform circular search pattern around waypoint"""
        try:
            self.logger.info(f"Performing search pattern (radius: {radius}m)")
            
            # Simple circular pattern
            num_points = 8
            for i in range(num_points):
                if not self.mission_active:
                    break
                
                angle = 2 * np.pi * i / num_points
                x = center_waypoint.x + radius * np.cos(angle)
                y = center_waypoint.y + radius * np.sin(angle)
                
                search_waypoint = Waypoint(x=x, y=y, z=center_waypoint.z)
                self._fly_to_waypoint(search_waypoint)
                
                # Hover and scan
                time.sleep(3)
            
            # Return to center
            self._fly_to_waypoint(center_waypoint)
            
        except Exception as e:
            self.logger.error(f"Search pattern error: {e}")
    
    def publish_mission_status(self):
        """Publish mission status"""
        try:
            status = {
                'mission_type': self.current_mission.value if self.current_mission else None,
                'active': self.mission_active,
                'current_waypoint': self.current_waypoint_index,
                'total_waypoints': len(self.mission_waypoints),
                'current_position': self.current_position.tolist(),
                'priority_detections': len(self.priority_detections),
                'detected_obstacles': len(self.detected_obstacles),
                'timestamp': time.time()
            }
            
            self.mission_status_pub(status)
            
        except Exception as e:
            self.logger.error(f"Mission status publish error: {e}")

# ============================================================================
# MAIN COORDINATOR
# ============================================================================

class DroneCoordinator(BaseNode):
    """Main coordinator for the disaster response drone system"""
    
    def __init__(self):
        super().__init__("drone_coordinator")
        
        # Initialize subsystems
        self.mavlink_interface = MAVLinkInterface()
        self.yolo_perception = YOLOPerception()
        self.realsense_processor = RealSenseProcessor()
        self.lidar_processor = LiDARProcessor()
        self.slam_processor = SLAMProcessor()
        self.navigation_controller = NavigationController()
        self.payload_controller = PayloadController()
        self.mission_executor = MissionExecutor()
        
        # State
        self.current_state = DroneState.INITIALIZING
        self.system_ready = False
        
        # Subscribers for system monitoring
        self.create_subscription("mavlink/state", dict, self.state_callback)
        self.create_subscription("mission/status", dict, self.mission_status_callback)
        
        # Publishers
        self.status_pub = self.create_publisher("system/status", dict)
        
        # Create status timer
        self.create_timer(1.0, self.publish_status)
        
        self.logger.info("Drone coordinator initialized")
        
        # Initialize system
        self.start_thread(self.initialization_sequence)
    
    def state_callback(self, msg):
        """Handle MAVLink state updates"""
        try:
            if msg.get('connected', False) and not self.system_ready:
                if self.current_state == DroneState.INITIALIZING:
                    self.current_state = DroneState.IDLE
                    self.system_ready = True
                    self.logger.info("Drone system ready")
        except Exception as e:
            self.logger.error(f"State callback error: {e}")
    
    def mission_status_callback(self, msg):
        """Handle mission status updates"""
        try:
            if msg.get('active', False):
                if self.current_state not in [DroneState.NAVIGATION, DroneState.PAYLOAD_DELIVERY]:
                    self.current_state = DroneState.NAVIGATION
        except Exception as e:
            self.logger.error(f"Mission status callback error: {e}")
    
    def publish_status(self):
        """Publish system status"""
        try:
            # Get latest sensor data
            gps = self.message_bus.get_latest_message("mavlink/gps")
            battery = self.message_bus.get_latest_message("mavlink/battery")
            detections = self.message_bus.get_latest_message("perception/detections")
            mission_status = self.message_bus.get_latest_message("mission/status")
            
            status = {
                'timestamp': time.time(),
                'state': self.current_state.value,
                'system_ready': self.system_ready,
                'subsystems': {
                    'mavlink': self.mavlink_interface.connected,
                    'perception': self.yolo_perception.active,
                    'camera': self.realsense_processor.active,
                    'lidar': self.lidar_processor.active,
                    'slam': self.slam_processor.active,
                    'navigation': self.navigation_controller.active,
                    'payload': self.payload_controller.active,
                    'mission': self.mission_executor.active
                },
                'telemetry': {
                    'gps_available': gps is not None,
                    'battery_voltage': battery.voltage if battery else 0.0,
                    'battery_percentage': battery.percentage if battery else 0.0,
                    'detection_count': len(detections) if detections else 0,
                    'mission_active': mission_status.get('active', False) if mission_status else False
                }
            }
            
            self.status_pub(status)
            
        except Exception as e:
            self.logger.error(f"Status publish error: {e}")
    
    def initialization_sequence(self):
        """Initialize the drone system"""
        try:
            self.logger.info("Starting initialization sequence...")
            
            # Wait for MAVLink connection
            timeout = 30.0
            start_time = time.time()
            while not self.mavlink_interface.connected and (time.time() - start_time) < timeout:
                self.logger.info("Waiting for MAVLink connection...")
                time.sleep(1.0)
            
            if not self.mavlink_interface.connected:
                self.logger.error("Failed to connect to autopilot")
                return
            
            # Set to GUIDED mode
            result = self.call_service("mavlink/set_mode", {'mode': 'GUIDED'})
            if result.get('success'):
                self.logger.info("Set to GUIDED mode")
            else:
                self.logger.warning(f"Failed to set GUIDED mode: {result.get('message')}")
            
            # Wait for all subsystems
            time.sleep(5)
            
            self.logger.info("Initialization complete")
            self.current_state = DroneState.IDLE
            self.system_ready = True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
    
    def arm_and_takeoff(self, altitude: float = 10.0):
        """Arm and takeoff to specified altitude"""
        try:
            if not self.system_ready:
                self.logger.error("System not ready")
                return False
            
            # Arm the drone
            result = self.call_service("mavlink/arm", {'arm': True})
            if not result.get('success'):
                self.logger.error(f"Failed to arm: {result.get('message')}")
                return False
            
            self.current_state = DroneState.ARMED
            self.logger.info("Drone armed")
            
            # Takeoff
            result = self.call_service("mavlink/takeoff", {'altitude': altitude})
            if not result.get('success'):
                self.logger.error(f"Failed to takeoff: {result.get('message')}")
                return False
            
            self.current_state = DroneState.TAKEOFF
            self.logger.info(f"Taking off to {altitude}m")
            
            # Wait for takeoff completion
            start_time = time.time()
            while (time.time() - start_time) < 30:  # 30 second timeout
                gps = self.message_bus.get_latest_message("mavlink/gps")
                if gps and gps.altitude >= altitude * 0.9:
                    self.current_state = DroneState.NAVIGATION
                    self.logger.info(f"Takeoff complete, altitude: {gps.altitude:.1f}m")
                    return True
                time.sleep(1.0)
            
            self.logger.warning("Takeoff timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Arm and takeoff error: {e}")
            return False
    
    def land(self):
        """Land the drone"""
        try:
            result = self.call_service("mavlink/land", {})
            if result.get('success'):
                self.current_state = DroneState.LANDING
                self.logger.info("Landing")
                return True
            else:
                self.logger.error(f"Failed to land: {result.get('message')}")
                return False
        except Exception as e:
            self.logger.error(f"Land error: {e}")
            return False
    
    def start_search_rescue_mission(self, waypoints: List[Dict], altitude: float = 15.0):
        """Start a search and rescue mission"""
        try:
            result = self.call_service("mission/start_search_rescue", {
                'waypoints': waypoints,
                'altitude': altitude
            })
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Start search rescue mission error: {e}")
            return False
    
    def start_delivery_mission(self, target: Dict, payload_weight: float = 2.0):
        """Start a delivery mission"""
        try:
            result = self.call_service("mission/start_delivery", {
                'target': target,
                'payload_weight': payload_weight
            })
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Start delivery mission error: {e}")
            return False
    
    def stop_mission(self):
        """Stop current mission"""
        try:
            result = self.call_service("mission/stop", {})
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Stop mission error: {e}")
            return False
    
    def shutdown_system(self):
        """Shutdown all subsystems"""
        self.logger.info("Shutting down drone system...")
        
        # Stop any active missions
        self.stop_mission()
        
        # Land if airborne
        if self.current_state in [DroneState.TAKEOFF, DroneState.NAVIGATION, DroneState.ARMED]:
            self.land()
            time.sleep(5)
        
        # Shutdown nodes
        nodes = [
            self.mavlink_interface, self.yolo_perception, self.realsense_processor,
            self.lidar_processor, self.slam_processor, self.navigation_controller,
            self.payload_controller, self.mission_executor
        ]
        
        for node in nodes:
            node.shutdown()
        
        # Shutdown message bus
        message_bus.shutdown()
        
        self.shutdown()
        self.logger.info("System shutdown complete")

# ============================================================================
# EXAMPLE USAGE AND DEMO
# ============================================================================

class SimpleMissionDemo:
    """Simple demo showing system capabilities"""
    
    def __init__(self, coordinator: DroneCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
    
    def run_basic_flight_demo(self):
        """Run basic flight demonstration"""
        try:
            self.logger.info("=== Basic Flight Demo ===")
            
            # Wait for system ready
            timeout = 30
            start_time = time.time()
            while not self.coordinator.system_ready and (time.time() - start_time) < timeout:
                self.logger.info("Waiting for system to be ready...")
                time.sleep(2)
            
            if not self.coordinator.system_ready:
                self.logger.error("System not ready, aborting demo")
                return False
            
            # Takeoff
            self.logger.info("Taking off to 10m...")
            if not self.coordinator.arm_and_takeoff(10.0):
                self.logger.error("Takeoff failed")
                return False
            
            # Hover for a bit
            self.logger.info("Hovering for 10 seconds...")
            time.sleep(10)
            
            # Test navigation
            self.logger.info("Testing navigation to position (10, 10, 10)")
            self.coordinator.navigation_controller.fly_to_position(10, 10, 10, 0)
            time.sleep(15)
            
            # Return to start
            self.logger.info("Returning to start position")
            self.coordinator.navigation_controller.fly_to_position(0, 0, 10, 0)
            time.sleep(15)
            
            # Land
            self.logger.info("Landing...")
            self.coordinator.land()
            time.sleep(10)
            
            self.logger.info("Basic flight demo completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Basic flight demo error: {e}")
            return False
    
    def run_search_rescue_demo(self):
        """Run search and rescue demonstration"""
        try:
            self.logger.info("=== Search and Rescue Demo ===")
            
            # Define search area waypoints
            search_waypoints = [
                {'x': 20, 'y': 20},
                {'x': 40, 'y': 20}, 
                {'x': 40, 'y': 40},
                {'x': 20, 'y': 40}
            ]
            
            # Takeoff
            if not self.coordinator.arm_and_takeoff(15.0):
                self.logger.error("Takeoff failed")
                return False
            
            # Start search mission
            self.logger.info("Starting search and rescue mission")
            if not self.coordinator.start_search_rescue_mission(search_waypoints, 15.0):
                self.logger.error("Failed to start search mission")
                return False
            
            # Monitor mission
            mission_timeout = 300  # 5 minutes
            start_time = time.time()
            
            while (time.time() - start_time) < mission_timeout:
                mission_status = self.coordinator.message_bus.get_latest_message("mission/status")
                if mission_status and not mission_status.get('active', False):
                    break
                
                # Log status
                if mission_status:
                    current_wp = mission_status.get('current_waypoint', 0)
                    total_wp = mission_status.get('total_waypoints', 0)
                    self.logger.info(f"Mission progress: {current_wp}/{total_wp}")
                
                time.sleep(10)
            
            # Land
            self.coordinator.land()
            time.sleep(10)
            
            self.logger.info("Search and rescue demo completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Search rescue demo error: {e}")
            return False
    
    def run_delivery_demo(self):
        """Run delivery demonstration"""
        try:
            self.logger.info("=== Delivery Demo ===")
            
            # Define delivery target
            delivery_target = {'x': 30, 'y': 40, 'z': 25}
            payload_weight = 2.0
            
            # Takeoff
            if not self.coordinator.arm_and_takeoff(20.0):
                self.logger.error("Takeoff failed")
                return False
            
            # Start delivery mission
            self.logger.info(f"Starting delivery mission to {delivery_target}")
            if not self.coordinator.start_delivery_mission(delivery_target, payload_weight):
                self.logger.error("Failed to start delivery mission")
                return False
            
            # Monitor mission
            mission_timeout = 300  # 5 minutes
            start_time = time.time()
            
            while (time.time() - start_time) < mission_timeout:
                mission_status = self.coordinator.message_bus.get_latest_message("mission/status")
                if mission_status and not mission_status.get('active', False):
                    break
                
                time.sleep(5)
            
            # Return and land
            self.coordinator.land()
            time.sleep(10)
            
            self.logger.info("Delivery demo completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Delivery demo error: {e}")
            return False

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the disaster response drone system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('drone_system.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting Disaster Response Hexacopter System - Pure Python")
    logger.info("=" * 80)
    
    try:
        # Create main coordinator
        coordinator = DroneCoordinator()
        
        # Wait for initialization
        logger.info("System initializing...")
        time.sleep(10)
        
        # Create demo runner
        demo = SimpleMissionDemo(coordinator)
        
        # Run demos based on command line or interactive choice
        import sys
        if len(sys.argv) > 1:
            demo_type = sys.argv[1].lower()
            
            if demo_type == "basic":
                demo.run_basic_flight_demo()
            elif demo_type == "search":
                demo.run_search_rescue_demo()
            elif demo_type == "delivery":
                demo.run_delivery_demo()
            else:
                logger.info("Unknown demo type. Available: basic, search, delivery")
        else:
            # Interactive mode
            logger.info("System ready. Available demos:")
            logger.info("1. Basic flight test")
            logger.info("2. Search and rescue mission")
            logger.info("3. Delivery mission")
            logger.info("4. System monitoring only")
            
            try:
                choice = input("Enter choice (1-4) or 'q' to quit: ")
                
                if choice == '1':
                    demo.run_basic_flight_demo()
                elif choice == '2':
                    demo.run_search_rescue_demo()
                elif choice == '3':
                    demo.run_delivery_demo()
                elif choice == '4':
                    logger.info("Monitoring system. Press Ctrl+C to stop.")
                    while True:
                        time.sleep(5)
                        status = coordinator.message_bus.get_latest_message("system/status")
                        if status:
                            logger.info(f"System Status: {status['state']}")
                elif choice.lower() == 'q':
                    logger.info("Exiting...")
                else:
                    logger.info("Invalid choice")
                    
            except EOFError:
                logger.info("Running system monitoring...")
                while True:
                    time.sleep(5)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        if 'coordinator' in locals():
            coordinator.shutdown_system()
        logger.info("System stopped")

if __name__ == "__main__":
    main()