import numpy as np
import cv2
import time
import threading
import json
import logging
import math
import heapq
import random
from typing import List, Tuple, Dict, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, PriorityQueue
from collections import deque
from scipy.spatial import distance

# Optional imports with fallbacks
try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
    MAVLINK_AVAILABLE = True
except ImportError:
    print("PyMAVLink not available - running in simulation mode")
    MAVLINK_AVAILABLE = False
    mavutil = None
    mavlink = None

# ============================================================================
# DATA CLASSES AND ENUMS
# ============================================================================

@dataclass
class LocalPoint:
    """Local coordinate point (relative to starting position)"""
    x: float  # meters from start position
    y: float  # meters from start position  
    z: float = 5.0  # altitude in meters

    def distance_to(self, other: 'LocalPoint') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def __eq__(self, other):
        return abs(self.x - other.x) < 0.5 and abs(self.y - other.y) < 0.5 and abs(self.z - other.z) < 1.0

    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1), round(self.z, 1)))

@dataclass
class VisualPose:
    """Visual pose estimation without GPS coordinates"""
    x: float = 0.0  # Local coordinate system (meters from start)
    y: float = 0.0  # Local coordinate system (meters from start) 
    z: float = 0.0  # Altitude (meters)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0

@dataclass
class VisualLandmark:
    """Visual landmark for navigation"""
    id: int
    position: np.ndarray  # 3D position in local coordinates
    descriptor: np.ndarray  # Visual descriptor (ORB features)
    keypoints: List[cv2.KeyPoint]
    observations: int = 0
    last_seen: float = 0.0
    landmark_type: str = "unknown"  # building, tree, vehicle, etc.

@dataclass
class CameraObstacle:
    """Obstacle detected by camera"""
    bearing: float  # angle from camera center (radians)
    elevation: float  # elevation angle (radians)
    distance: float  # estimated distance (meters)
    size: Tuple[float, float]  # width, height in image
    confidence: float  # detection confidence
    obstacle_type: str  # person, vehicle, building, etc.
    timestamp: float

@dataclass
class SensorObstacle:
    """Generic obstacle from any sensor"""
    x: float  # position relative to drone (meters)
    y: float  # position relative to drone (meters)
    z: float  # position relative to drone (meters)
    radius: float  # obstacle radius (meters)
    confidence: float
    sensor_type: str  # camera, lidar, ultrasonic
    velocity: Tuple[float, float] = (0.0, 0.0)  # for moving obstacles
    timestamp: float = 0.0

@dataclass
class VisionDroneState:
    """Drone state using vision-based navigation (no GPS)"""
    # Local coordinates (meters from takeoff point)
    local_x: float = 0.0
    local_y: float = 0.0
    altitude: float = 0.0

    # Orientation
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Motion
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    # System status
    battery_voltage: float = 0.0
    battery_current: float = 0.0
    battery_remaining: float = 100.0
    flight_mode: str = "UNKNOWN"
    armed: bool = False
    system_status: str = "UNKNOWN"

    # Vision system status
    visual_tracking_quality: float = 0.0
    landmarks_tracked: int = 0
    altitude_source: str = "barometer"  # barometer, lidar, camera

    # Navigation confidence
    position_confidence: float = 0.0
    heading_confidence: float = 0.0

    timestamp: float = 0.0

class MissionType(Enum):
    """Types of disaster response missions"""
    SEARCH_AND_RESCUE = "search_rescue"
    MEDICAL_DELIVERY = "medical_delivery"
    COMMUNICATION_RELAY = "communication_relay"
    DAMAGE_ASSESSMENT = "damage_assessment"
    EVACUATION_SUPPORT = "evacuation_support"

class MissionStatus(Enum):
    """Mission execution status"""
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"

class Priority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

# ============================================================================
# VISION NAVIGATION CLASSES
# ============================================================================

class VisualOdometry:
    """Monocular visual odometry system"""

    def __init__(self):
        # Feature detector and matcher
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Camera parameters (should be calibrated)
        self.camera_matrix = np.array([
            [500, 0, 320],  # fx, 0, cx
            [0, 500, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4, 1))

        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Pose estimation
        self.current_pose = VisualPose()
        self.pose_history = deque(maxlen=1000)

        # Scale estimation (since monocular)
        self.scale_factor = 1.0
        self.altitude_estimate = 0.0

    def process_frame(self, frame: np.ndarray, altitude_hint: float = None) -> VisualPose:
        """Process new frame and estimate pose"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        if self.prev_frame is not None and self.prev_descriptors is not None:
            # Match features
            matches = self._match_features(self.prev_descriptors, descriptors)

            if len(matches) > 50:  # Sufficient matches for pose estimation
                # Extract matched points
                prev_pts, curr_pts = self._extract_matched_points(
                    self.prev_keypoints, keypoints, matches
                )

                # Estimate pose change
                pose_change = self._estimate_pose_change(prev_pts, curr_pts, altitude_hint)

                # Update current pose
                self._update_pose(pose_change)

        # Update previous frame data
        self.prev_frame = gray.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        self.current_pose.timestamp = time.time()
        self.pose_history.append(self.current_pose)

        return self.current_pose

    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Match features between frames"""
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.match(desc1, desc2)

        # Filter good matches
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:100] if len(matches) > 100 else matches

        return good_matches

    def _extract_matched_points(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                               matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract matched point coordinates"""
        prev_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        return prev_pts.reshape(-1, 1, 2), curr_pts.reshape(-1, 1, 2)

    def _estimate_pose_change(self, prev_pts: np.ndarray, curr_pts: np.ndarray, 
                             altitude_hint: float = None) -> Dict[str, float]:
        """Estimate pose change between frames"""
        try:
            # Calculate essential matrix
            E, mask = cv2.findEssentialMat(
                prev_pts, curr_pts, self.camera_matrix,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            if E is not None:
                # Recover pose from essential matrix
                _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix)

                # Estimate scale from altitude if available
                if altitude_hint is not None:
                    self.altitude_estimate = altitude_hint
                    # Simple scale estimation based on altitude
                    self.scale_factor = max(altitude_hint / 10.0, 0.1)  # Rough heuristic

                # Extract translation and rotation
                translation = t.flatten() * self.scale_factor

                # Convert rotation matrix to Euler angles
                rotation = self._rotation_matrix_to_euler(R)

                return {
                    'dx': translation[0],
                    'dy': translation[1], 
                    'dz': translation[2],
                    'droll': rotation[0],
                    'dpitch': rotation[1],
                    'dyaw': rotation[2],
                    'confidence': len(prev_pts[mask.ravel() == 1]) / max(len(prev_pts), 1)
                }
        except Exception as e:
            pass

        return {
            'dx': 0, 'dy': 0, 'dz': 0,
            'droll': 0, 'dpitch': 0, 'dyaw': 0,
            'confidence': 0.0
        }

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles"""
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    def _update_pose(self, pose_change: Dict[str, float]):
        """Update current pose with estimated change"""
        # Apply rotation first (body frame)
        cos_yaw = math.cos(self.current_pose.yaw)
        sin_yaw = math.sin(self.current_pose.yaw)

        # Transform translation to world frame
        dx_world = pose_change['dx'] * cos_yaw - pose_change['dy'] * sin_yaw
        dy_world = pose_change['dx'] * sin_yaw + pose_change['dy'] * cos_yaw

        # Update position
        self.current_pose.x += dx_world
        self.current_pose.y += dy_world
        self.current_pose.z += pose_change['dz']

        # Update orientation
        self.current_pose.roll += pose_change['droll']
        self.current_pose.pitch += pose_change['dpitch'] 
        self.current_pose.yaw += pose_change['dyaw']

        # Normalize angles
        self.current_pose.yaw = math.atan2(
            math.sin(self.current_pose.yaw), 
            math.cos(self.current_pose.yaw)
        )

        self.current_pose.confidence = pose_change['confidence']

class VisualSLAM:
    """Visual SLAM system for mapping and localization"""

    def __init__(self):
        self.visual_odometry = VisualOdometry()
        self.landmarks = {}
        self.next_landmark_id = 0

        # Feature matching for landmarks
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Loop closure detection
        self.keyframes = []
        self.keyframe_interval = 2.0  # meters
        self.loop_closure_threshold = 5.0  # meters

        # Map optimization
        self.optimization_enabled = True

    def process_frame(self, frame: np.ndarray, altitude_hint: float = None) -> VisualPose:
        """Process frame and update SLAM"""
        # Get visual odometry pose
        pose = self.visual_odometry.process_frame(frame, altitude_hint)

        # Extract and match landmarks
        self._extract_landmarks(frame, pose)

        # Check for keyframe addition
        if self._should_add_keyframe(pose):
            self._add_keyframe(frame, pose)

            # Check for loop closure
            if self._detect_loop_closure():
                self._optimize_map()

        return pose

    def _extract_landmarks(self, frame: np.ndarray, pose: VisualPose):
        """Extract and track visual landmarks"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        if descriptors is None:
            return

        # Match with existing landmarks
        for landmark_id, landmark in self.landmarks.items():
            if landmark.descriptor.size == 0:
                continue

            matches = self.matcher.knnMatch(landmark.descriptor, descriptors, k=2)

            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 3:  # Landmark observed
                landmark.observations += 1
                landmark.last_seen = time.time()

        # Create new landmarks from unmatched features
        self._create_new_landmarks(keypoints, descriptors, pose)

    def _create_new_landmarks(self, keypoints: List[cv2.KeyPoint], 
                             descriptors: np.ndarray, pose: VisualPose):
        """Create new landmarks from detected features"""
        # Simple triangulation assumption for new landmarks
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            if i % 5 != 0:  # Only create landmark for every 5th feature
                continue

            # Estimate 3D position (simplified - assumes ground plane)
            bearing = self._pixel_to_bearing(kp.pt)
            distance = max(pose.z / abs(math.tan(bearing[1])), 1.0) if bearing[1] != 0 else 10.0

            # Transform to world coordinates
            world_pos = self._transform_to_world(
                np.array([distance * math.cos(bearing[0]), 
                         distance * math.sin(bearing[0]), 0]), 
                pose
            )

            # Create new landmark
            landmark = VisualLandmark(
                id=self.next_landmark_id,
                position=world_pos,
                descriptor=desc.copy(),
                keypoints=[kp],
                observations=1,
                last_seen=time.time(),
                landmark_type=self._classify_landmark(kp, desc)
            )

            self.landmarks[self.next_landmark_id] = landmark
            self.next_landmark_id += 1

            # Limit number of landmarks
            if len(self.landmarks) > 1000:
                self._remove_old_landmarks()

    def _pixel_to_bearing(self, pixel: Tuple[float, float]) -> np.ndarray:
        """Convert pixel coordinates to bearing angles"""
        camera_matrix = self.visual_odometry.camera_matrix

        fx, fy = camera_matrix[0,0], camera_matrix[1,1]
        cx, cy = camera_matrix[0,2], camera_matrix[1,2]

        x_norm = (pixel[0] - cx) / fx
        y_norm = (pixel[1] - cy) / fy

        azimuth = math.atan2(x_norm, 1.0)
        elevation = math.atan2(-y_norm, math.sqrt(x_norm**2 + 1.0))

        return np.array([azimuth, elevation])

    def _transform_to_world(self, local_pos: np.ndarray, pose: VisualPose) -> np.ndarray:
        """Transform local coordinates to world coordinates"""
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)

        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        world_pos = R @ local_pos + np.array([pose.x, pose.y, pose.z])
        return world_pos

    def _classify_landmark(self, keypoint: cv2.KeyPoint, descriptor: np.ndarray) -> str:
        """Simple landmark classification"""
        if keypoint.response > 0.01:
            return "building"
        elif keypoint.size > 20:
            return "tree"
        else:
            return "generic"

    def _should_add_keyframe(self, pose: VisualPose) -> bool:
        """Determine if current pose should be added as keyframe"""
        if not self.keyframes:
            return True

        last_keyframe = self.keyframes[-1]['pose']
        distance = math.sqrt(
            (pose.x - last_keyframe.x)**2 + 
            (pose.y - last_keyframe.y)**2
        )

        return distance > self.keyframe_interval

    def _add_keyframe(self, frame: np.ndarray, pose: VisualPose):
        """Add current frame as keyframe"""
        keyframe = {
            'pose': pose,
            'frame': frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
            'timestamp': time.time()
        }

        self.keyframes.append(keyframe)

        if len(self.keyframes) > 50:
            self.keyframes.pop(0)

    def _detect_loop_closure(self) -> bool:
        """Detect loop closure opportunities"""
        if len(self.keyframes) < 10:
            return False

        current_pose = self.keyframes[-1]['pose']

        for i, keyframe in enumerate(self.keyframes[:-10]):
            distance = math.sqrt(
                (current_pose.x - keyframe['pose'].x)**2 + 
                (current_pose.y - keyframe['pose'].y)**2
            )

            if distance < self.loop_closure_threshold:
                return True

        return False

    def _optimize_map(self):
        """Optimize map using loop closure"""
        if not self.optimization_enabled:
            return

        if len(self.keyframes) >= 2:
            correction_factor = 0.9

            for keyframe in self.keyframes[-10:]:
                keyframe['pose'].x *= correction_factor
                keyframe['pose'].y *= correction_factor

    def _remove_old_landmarks(self):
        """Remove old or rarely observed landmarks"""
        current_time = time.time()
        landmarks_to_remove = []

        for landmark_id, landmark in self.landmarks.items():
            if (current_time - landmark.last_seen > 30.0 or 
                landmark.observations < 3):
                landmarks_to_remove.append(landmark_id)

        for landmark_id in landmarks_to_remove[:100]:
            if landmark_id in self.landmarks:
                del self.landmarks[landmark_id]

    def get_current_pose(self) -> VisualPose:
        """Get current pose estimate"""
        return self.visual_odometry.current_pose

    def get_landmarks(self) -> Dict[int, VisualLandmark]:
        """Get current landmarks"""
        return self.landmarks.copy()

# ============================================================================
# PATH PLANNING CLASSES
# ============================================================================

@dataclass
class VisualWaypoint:
    """Waypoint defined by visual landmarks"""
    position: LocalPoint
    landmark_id: Optional[int] = None
    landmark_type: str = "generic"
    approach_direction: float = 0.0  # preferred approach angle
    accuracy_radius: float = 2.0  # acceptable distance to waypoint

class VisualObstacleMap:
    """Obstacle map based on visual detection"""

    def __init__(self, bounds: Tuple[float, float, float, float, float, float], resolution: float = 1.0):
        self.bounds = bounds
        self.resolution = resolution

        self.static_obstacles: Set[LocalPoint] = set()
        self.camera_obstacles: List[Tuple[LocalPoint, float, float]] = []
        self.landmark_obstacles: Dict[int, LocalPoint] = {}

        self.safety_margin = 2.0
        self.altitude_clearance = 3.0

    def add_camera_obstacle(self, obstacle_point: LocalPoint, confidence: float):
        """Add obstacle detected by camera"""
        current_time = time.time()
        self.camera_obstacles.append((obstacle_point, confidence, current_time))

        # Remove old camera obstacles
        self.camera_obstacles = [
            (pt, conf, ts) for pt, conf, ts in self.camera_obstacles 
            if current_time - ts < 5.0
        ]

    def is_obstacle(self, point: LocalPoint) -> Tuple[bool, str]:
        """Check if point is an obstacle"""
        if point in self.static_obstacles:
            return True, "static"

        for landmark_id, landmark_pos in self.landmark_obstacles.items():
            if point.distance_to(landmark_pos) < self.safety_margin:
                return True, "landmark"

        current_time = time.time()
        for obs_point, confidence, timestamp in self.camera_obstacles:
            if current_time - timestamp < 2.0 and confidence > 0.7:
                if point.distance_to(obs_point) < self.safety_margin:
                    return True, "camera"

        return False, "none"

    def is_within_bounds(self, point: LocalPoint) -> bool:
        """Check if point is within map bounds"""
        return (self.bounds[0] <= point.x <= self.bounds[1] and
                self.bounds[2] <= point.y <= self.bounds[3] and
                self.bounds[4] <= point.z <= self.bounds[5])

    def is_valid_point(self, point: LocalPoint) -> bool:
        """Check if point is valid for navigation"""
        return self.is_within_bounds(point) and not self.is_obstacle(point)[0]

    def get_neighbors(self, point: LocalPoint) -> List[LocalPoint]:
        """Get valid neighboring points"""
        neighbors = []
        directions = [
            (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
            (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0)
        ]

        for dx, dy, dz in directions:
            neighbor = LocalPoint(
                point.x + dx * self.resolution,
                point.y + dy * self.resolution,
                point.z + dz * self.resolution
            )

            if self.is_valid_point(neighbor):
                neighbors.append(neighbor)

        return neighbors

@dataclass
class NavigationNode:
    """Node for path planning"""
    point: LocalPoint
    parent: Optional['NavigationNode'] = None
    g_cost: float = 0.0
    h_cost: float = 0.0
    f_cost: float = 0.0

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class VisualPathPlanner:
    """A* path planner using visual landmarks"""

    def __init__(self, obstacle_map: VisualObstacleMap):
        self.obstacle_map = obstacle_map
        self.visual_landmarks = {}

    def add_visual_landmark(self, landmark_id: int, position: LocalPoint, landmark_type: str):
        """Add visual landmark for navigation"""
        self.visual_landmarks[landmark_id] = {
            'position': position,
            'type': landmark_type,
            'reliability': 1.0
        }

        if landmark_type in ["building", "tree", "pole", "wall"]:
            self.obstacle_map.landmark_obstacles[landmark_id] = position

    def plan_path(self, start: LocalPoint, goal: LocalPoint) -> Optional[List[LocalPoint]]:
        """Plan path using A* algorithm"""
        if not self.obstacle_map.is_valid_point(start) or not self.obstacle_map.is_valid_point(goal):
            return None

        open_set = []
        closed_set: Set[LocalPoint] = set()
        nodes = {}

        start_node = NavigationNode(start, None, 0.0, self.heuristic(start, goal))
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        heapq.heappush(open_set, start_node)
        nodes[start] = start_node

        while open_set:
            current_node = heapq.heappop(open_set)
            current_point = current_node.point

            if current_point in closed_set:
                continue

            closed_set.add(current_point)

            if current_point.distance_to(goal) < self.obstacle_map.resolution:
                return self._reconstruct_path(current_node)

            for neighbor_point in self.obstacle_map.get_neighbors(current_point):
                if neighbor_point in closed_set:
                    continue

                movement_cost = current_point.distance_to(neighbor_point)
                tentative_g = current_node.g_cost + movement_cost

                if neighbor_point not in nodes or tentative_g < nodes[neighbor_point].g_cost:
                    neighbor_node = NavigationNode(
                        neighbor_point,
                        current_node,
                        tentative_g,
                        self.heuristic(neighbor_point, goal)
                    )
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost

                    nodes[neighbor_point] = neighbor_node
                    heapq.heappush(open_set, neighbor_node)

        return None

    def heuristic(self, point1: LocalPoint, point2: LocalPoint) -> float:
        """Euclidean distance heuristic"""
        return point1.distance_to(point2)

    def _reconstruct_path(self, node: NavigationNode) -> List[LocalPoint]:
        """Reconstruct path from goal to start"""
        path = []
        current = node

        while current:
            path.append(current.point)
            current = current.parent

        return path[::-1]

# ============================================================================
# CAMERA AND OBSTACLE AVOIDANCE CLASSES
# ============================================================================

class CameraProcessor:
    """Advanced camera processing for obstacle detection"""

    def __init__(self):
        self.camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240], 
            [0, 0, 1]
        ], dtype=np.float32)

        self.image_width = 640
        self.image_height = 480
        self.fov_horizontal = 60.0  # degrees
        self.fov_vertical = 45.0    # degrees

        self.prev_frame = None

    def process_frame(self, frame: np.ndarray, altitude: float) -> List[CameraObstacle]:
        """Process camera frame to detect obstacles"""
        obstacles = []

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect obstacles using edge detection
        obstacles.extend(self._detect_edge_obstacles(gray, altitude))

        # Detect motion obstacles if we have previous frame
        if self.prev_frame is not None:
            obstacles.extend(self._detect_motion_obstacles(gray, altitude))

        self.prev_frame = gray.copy()

        return self._filter_obstacles(obstacles)

    def _detect_edge_obstacles(self, gray: np.ndarray, altitude: float) -> List[CameraObstacle]:
        """Detect obstacles using edge detection"""
        obstacles = []

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                bearing = self._pixel_to_bearing(center_x, self.image_width)
                elevation = self._pixel_to_elevation(center_y, self.image_height)
                estimated_distance = self._estimate_distance_from_size(w, h, altitude)

                obstacle = CameraObstacle(
                    bearing=bearing,
                    elevation=elevation,
                    distance=estimated_distance,
                    size=(w, h),
                    confidence=0.7,
                    obstacle_type="edge",
                    timestamp=time.time()
                )

                obstacles.append(obstacle)

        return obstacles

    def _detect_motion_obstacles(self, gray: np.ndarray, altitude: float) -> List[CameraObstacle]:
        """Detect moving obstacles"""
        obstacles = []

        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find moving contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                bearing = self._pixel_to_bearing(center_x, self.image_width)
                elevation = self._pixel_to_elevation(center_y, self.image_height)

                obstacle = CameraObstacle(
                    bearing=bearing,
                    elevation=elevation,
                    distance=max(altitude * 0.5, 3.0),
                    size=(w, h),
                    confidence=0.8,
                    obstacle_type="moving",
                    timestamp=time.time()
                )

                obstacles.append(obstacle)

        return obstacles

    def _pixel_to_bearing(self, pixel_x: float, image_width: float) -> float:
        """Convert pixel coordinate to bearing angle"""
        normalized_x = (pixel_x - image_width / 2) / (image_width / 2)
        max_angle = math.radians(self.fov_horizontal / 2)
        return normalized_x * max_angle

    def _pixel_to_elevation(self, pixel_y: float, image_height: float) -> float:
        """Convert pixel coordinate to elevation angle"""
        normalized_y = (image_height / 2 - pixel_y) / (image_height / 2)
        max_angle = math.radians(self.fov_vertical / 2)
        return normalized_y * max_angle

    def _estimate_distance_from_size(self, width: float, height: float, altitude: float) -> float:
        """Estimate distance based on object size"""
        object_size = max(width, height)

        if object_size > 100:
            return max(altitude * 0.3, 2.0)
        elif object_size > 50:
            return max(altitude * 0.6, 5.0)
        else:
            return max(altitude * 0.9, 10.0)

    def _filter_obstacles(self, obstacles: List[CameraObstacle]) -> List[CameraObstacle]:
        """Filter and merge similar obstacles"""
        if not obstacles:
            return obstacles

        filtered = []
        merge_threshold = 0.2

        obstacles.sort(key=lambda x: x.confidence, reverse=True)

        for obstacle in obstacles:
            should_add = True

            for existing in filtered:
                bearing_diff = abs(obstacle.bearing - existing.bearing)
                elevation_diff = abs(obstacle.elevation - existing.elevation)

                if bearing_diff < merge_threshold and elevation_diff < merge_threshold:
                    if obstacle.confidence > existing.confidence:
                        filtered.remove(existing)
                        filtered.append(obstacle)
                    should_add = False
                    break

            if should_add:
                filtered.append(obstacle)

        return filtered

class CameraEnhancedAvoidance:
    """Main camera-enhanced obstacle avoidance system"""

    def __init__(self):
        self.camera_processor = CameraProcessor()

        self.safety_distance = 5.0
        self.max_avoidance_angle = math.pi / 3

        self.obstacle_buffer = deque(maxlen=50)

    def process_camera_frame(self, frame: np.ndarray, altitude: float, 
                           current_heading: float) -> Tuple[float, float, bool]:
        """Process camera frame and return avoidance commands"""
        camera_obstacles = self.camera_processor.process_frame(frame, altitude)

        sensor_obstacles = self._convert_camera_obstacles(camera_obstacles, current_heading)

        current_time = time.time()
        for obs in sensor_obstacles:
            obs.timestamp = current_time
            self.obstacle_buffer.append(obs)

        self._clean_old_obstacles()

        return self._calculate_avoidance_command(current_heading)

    def _convert_camera_obstacles(self, camera_obstacles: List[CameraObstacle], 
                                 current_heading: float) -> List[SensorObstacle]:
        """Convert camera obstacles to sensor obstacles"""
        sensor_obstacles = []

        for cam_obs in camera_obstacles:
            x = cam_obs.distance * math.sin(cam_obs.bearing)
            y = cam_obs.distance * math.cos(cam_obs.bearing) * math.cos(cam_obs.elevation)
            z = -cam_obs.distance * math.sin(cam_obs.elevation)

            radius = max(cam_obs.size[0], cam_obs.size[1]) / 100.0 + 1.0

            sensor_obs = SensorObstacle(
                x=x, y=y, z=z,
                radius=radius,
                confidence=cam_obs.confidence,
                sensor_type="camera",
                timestamp=cam_obs.timestamp
            )

            sensor_obstacles.append(sensor_obs)

        return sensor_obstacles

    def _clean_old_obstacles(self):
        """Remove obstacles older than 2 seconds"""
        current_time = time.time()
        fresh_obstacles = deque(maxlen=50)

        for obs in self.obstacle_buffer:
            if current_time - obs.timestamp < 2.0:
                fresh_obstacles.append(obs)

        self.obstacle_buffer = fresh_obstacles

    def _calculate_avoidance_command(self, current_heading: float) -> Tuple[float, float, bool]:
        """Calculate avoidance steering and speed commands"""
        if not self.obstacle_buffer:
            return 0.0, 1.0, False

        closest_obstacle = None
        min_distance = float('inf')

        for obs in self.obstacle_buffer:
            distance = math.sqrt(obs.x**2 + obs.y**2 + obs.z**2)
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obs

        if min_distance < 2.0 and closest_obstacle.confidence > 0.7:
            return 0.0, 0.0, True  # Emergency stop

        total_avoidance_x = 0.0
        total_avoidance_y = 0.0
        total_weight = 0.0

        for obs in self.obstacle_buffer:
            distance = math.sqrt(obs.x**2 + obs.y**2)

            if distance < self.safety_distance and distance > 0:
                weight = obs.confidence / (distance**2)

                avoidance_x = -obs.x / distance * weight
                avoidance_y = -obs.y / distance * weight

                total_avoidance_x += avoidance_x
                total_avoidance_y += avoidance_y
                total_weight += weight

        if total_weight == 0:
            return 0.0, 1.0, False

        avg_avoidance_x = total_avoidance_x / total_weight
        avg_avoidance_y = total_avoidance_y / total_weight

        avoidance_angle = math.atan2(avg_avoidance_x, avg_avoidance_y)

        steering_angle = max(-self.max_avoidance_angle, 
                           min(self.max_avoidance_angle, avoidance_angle))

        avoidance_magnitude = math.sqrt(avg_avoidance_x**2 + avg_avoidance_y**2)
        speed_factor = max(0.3, 1.0 - avoidance_magnitude * 0.5)

        return steering_angle, speed_factor, False

# ============================================================================
# MAVLINK INTEGRATION CLASSES
# ============================================================================

class VisionMAVLinkConnection:
    """MAVLink connection using vision-based navigation"""

    def __init__(self, connection_string: str = "/dev/ttyUSB0", baudrate: int = 57600):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.master = None
        self.connected = False
        self.running = False

        self.drone_state = VisionDroneState()
        self.state_lock = threading.Lock()

        self.home_position = LocalPoint(0, 0, 0)

        self.incoming_messages = Queue()
        self.outgoing_messages = Queue()
        self.message_callbacks: Dict[str, List[Callable]] = {}

        self.read_thread = None
        self.write_thread = None

    def connect(self) -> bool:
        """Connect to drone and initialize vision system"""
        if not MAVLINK_AVAILABLE:
            print("MAVLink not available - running in simulation mode")
            self.connected = True
            self.running = True
            self._start_simulation_mode()
            return True

        try:
            self.master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baudrate,
                source_system=255
            )

            self.master.wait_heartbeat()

            self.connected = True
            self.running = True

            self.read_thread = threading.Thread(target=self._read_messages, daemon=True)
            self.write_thread = threading.Thread(target=self._write_messages, daemon=True)

            self.read_thread.start()
            self.write_thread.start()

            self._configure_vision_mode()

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def _start_simulation_mode(self):
        """Start simulation mode for testing"""
        def simulate_drone():
            while self.running:
                with self.state_lock:
                    # Simulate basic drone state
                    self.drone_state.battery_remaining = max(0, self.drone_state.battery_remaining - 0.1)
                    self.drone_state.flight_mode = "GUIDED"
                    self.drone_state.visual_tracking_quality = 0.85
                    self.drone_state.landmarks_tracked = 5
                    self.drone_state.timestamp = time.time()

                time.sleep(1.0)

        sim_thread = threading.Thread(target=simulate_drone, daemon=True)
        sim_thread.start()

    def _configure_vision_mode(self):
        """Configure drone for vision-based navigation"""
        if not MAVLINK_AVAILABLE:
            return

        # Configure parameters for vision navigation
        params_to_set = [
            (b'GPS_TYPE', 0),  # Disable GPS
            (b'VISO_TYPE', 1),  # Enable visual odometry
            (b'EK3_SRC1_POSXY', 0),  # No position aiding
            (b'EK3_SRC1_POSZ', 1),  # Barometer for altitude
        ]

        for param_name, param_value in params_to_set:
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                param_name,
                param_value,
                mavlink.MAV_PARAM_TYPE_UINT8
            )

    def _read_messages(self):
        """Read messages from drone"""
        while self.running and self.connected:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                if msg is not None:
                    self.incoming_messages.put(msg)
                    self._process_message(msg)
            except Exception as e:
                time.sleep(0.1)

    def _write_messages(self):
        """Send messages to drone"""
        while self.running and self.connected:
            try:
                if not self.outgoing_messages.empty():
                    msg = self.outgoing_messages.get(timeout=1.0)
                    if MAVLINK_AVAILABLE:
                        self.master.mav.send(msg)
            except Exception as e:
                time.sleep(0.1)

    def _process_message(self, msg):
        """Process incoming MAVLink message"""
        with self.state_lock:
            msg_type = msg.get_type()

            if msg_type == 'HEARTBEAT':
                if MAVLINK_AVAILABLE:
                    self.drone_state.flight_mode = mavutil.mode_string_v10(msg)
                    self.drone_state.armed = (msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

            elif msg_type == 'SYS_STATUS':
                self.drone_state.battery_voltage = msg.voltage_battery / 1000.0
                self.drone_state.battery_current = msg.current_battery / 100.0
                self.drone_state.battery_remaining = msg.battery_remaining

            elif msg_type == 'LOCAL_POSITION_NED':
                self.drone_state.local_x = msg.x
                self.drone_state.local_y = msg.y
                self.drone_state.altitude = -msg.z

            self.drone_state.timestamp = time.time()

        if msg_type in self.message_callbacks:
            for callback in self.message_callbacks[msg_type]:
                try:
                    callback(msg)
                except:
                    pass

    def get_drone_state(self) -> VisionDroneState:
        """Get current drone state (thread-safe)"""
        with self.state_lock:
            return VisionDroneState(
                local_x=self.drone_state.local_x,
                local_y=self.drone_state.local_y,
                altitude=self.drone_state.altitude,
                roll=self.drone_state.roll,
                pitch=self.drone_state.pitch,
                yaw=self.drone_state.yaw,
                velocity_x=self.drone_state.velocity_x,
                velocity_y=self.drone_state.velocity_y,
                velocity_z=self.drone_state.velocity_z,
                battery_voltage=self.drone_state.battery_voltage,
                battery_current=self.drone_state.battery_current,
                battery_remaining=self.drone_state.battery_remaining,
                flight_mode=self.drone_state.flight_mode,
                armed=self.drone_state.armed,
                system_status=self.drone_state.system_status,
                visual_tracking_quality=self.drone_state.visual_tracking_quality,
                landmarks_tracked=self.drone_state.landmarks_tracked,
                position_confidence=self.drone_state.position_confidence,
                heading_confidence=self.drone_state.heading_confidence,
                timestamp=self.drone_state.timestamp
            )

class VisionDroneCommands:
    """High-level drone commands for vision-based navigation"""

    def __init__(self, mavlink_connection: VisionMAVLinkConnection):
        self.conn = mavlink_connection

    def arm_drone(self) -> bool:
        """Arm the drone"""
        if not self.conn.connected or not MAVLINK_AVAILABLE:
            print("Arm command sent (simulation mode)")
            return True

        msg = self.conn.master.mav.command_long_encode(
            self.conn.master.target_system,
            self.conn.master.target_component,
            mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

        self.conn.outgoing_messages.put(msg)
        return True

    def takeoff(self, altitude: float) -> bool:
        """Takeoff to specified altitude"""
        if not self.conn.connected:
            return False

        print(f"Takeoff to {altitude}m (vision-based navigation)")

        if not MAVLINK_AVAILABLE:
            return True

        msg = self.conn.master.mav.command_long_encode(
            self.conn.master.target_system,
            self.conn.master.target_component,
            mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

        self.conn.outgoing_messages.put(msg)
        return True

    def goto_local_position(self, x: float, y: float, z: float) -> bool:
        """Go to local position"""
        if not self.conn.connected:
            return False

        print(f"Moving to local position: ({x:.1f}, {y:.1f}, {z:.1f})")

        if not MAVLINK_AVAILABLE:
            return True

        msg = self.conn.master.mav.set_position_target_local_ned_encode(
            0, self.conn.master.target_system, self.conn.master.target_component,
            mavlink.MAV_FRAME_LOCAL_NED, 0b0000111111111000,
            x, y, -z, 0, 0, 0, 0, 0, 0, 0, 0
        )

        self.conn.outgoing_messages.put(msg)
        return True

# ============================================================================
# MAIN VISION-BASED DISASTER RESPONSE SYSTEM
# ============================================================================

@dataclass
class VisionMission:
    """Vision-based mission definition"""
    mission_id: str
    mission_type: MissionType
    priority: Priority
    target_location: LocalPoint
    waypoints: List[LocalPoint]
    payload_type: str
    estimated_duration: float
    created_timestamp: float

class VisionBasedDisasterSystem:
    """Main vision-based disaster response system"""

    def __init__(self, config_file: str = "vision_config.json"):
        self.config = self._load_config(config_file)
        self._setup_logging()

        # Initialize vision-based subsystems
        self.visual_slam = VisualSLAM()
        self.obstacle_map = VisualObstacleMap((-50, 50, -50, 50, 0, 30), resolution=1.0)
        self.path_planner = VisualPathPlanner(self.obstacle_map)
        self.obstacle_avoidance = CameraEnhancedAvoidance()

        # Initialize communication
        self.mavlink_conn = VisionMAVLinkConnection(
            self.config.get("mavlink_connection", "/dev/ttyUSB0"),
            self.config.get("baudrate", 57600)
        )
        self.drone_commands = VisionDroneCommands(self.mavlink_conn)

        # System state
        self.current_mission: Optional[VisionMission] = None
        self.mission_status = MissionStatus.PLANNED
        self.current_position = LocalPoint(0, 0, 10)
        self.home_position = LocalPoint(0, 0, 0)

        # Camera interface
        self.camera_capture = None
        self.camera_active = False

        # Task management
        self.task_queue = PriorityQueue()
        self.completed_missions = []

        # Performance metrics
        self.performance_metrics = {
            "missions_completed": 0,
            "total_flight_time": 0.0,
            "visual_navigation_accuracy": 0.0,
            "landmarks_mapped": 0,
            "obstacles_avoided": 0
        }

        # Threading
        self.running = False
        self.main_thread = None
        self.camera_thread = None

        # System health
        self.system_health = {
            "overall": "healthy",
            "communication": "connected",
            "cameras": "operational",
            "visual_tracking": "good",
            "battery": 100
        }

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "mavlink_connection": "/dev/ttyUSB0",
            "baudrate": 57600,
            "camera_device": 0,
            "max_altitude": 50,
            "max_speed": 10,
            "visual_quality_threshold": 0.7,
            "camera_fps": 30
        }

        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                return {**default_config, **user_config}
        except FileNotFoundError:
            return default_config

    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vision_drone_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('VisionDrone')

    def start_system(self) -> bool:
        """Start the vision-based system"""
        self.logger.info("Starting Vision-Based Disaster Response System")

        # Initialize camera
        if not self._initialize_camera():
            self.logger.warning("Camera initialization failed - continuing without camera")

        # Connect to drone
        if not self.mavlink_conn.connect():
            self.logger.error("Failed to connect to drone")
            return False

        try:
            self.running = True

            # Start processing threads
            self.main_thread = threading.Thread(target=self._main_control_loop, daemon=True)
            if self.camera_active:
                self.camera_thread = threading.Thread(target=self._camera_processing_loop, daemon=True)
                self.camera_thread.start()

            self.main_thread.start()

            self.logger.info("Vision-based system started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            return False

    def _initialize_camera(self) -> bool:
        """Initialize camera system"""
        try:
            camera_device = self.config.get("camera_device", 0)
            self.camera_capture = cv2.VideoCapture(camera_device)

            if not self.camera_capture.isOpened():
                return False

            # Set camera properties
            self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_capture.set(cv2.CAP_PROP_FPS, self.config.get("camera_fps", 30))

            self.camera_active = True
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def _main_control_loop(self):
        """Main control loop"""
        while self.running:
            try:
                # Update system health
                self._update_system_health()

                # Process mission queue
                self._process_mission_queue()

                # Monitor active mission
                if self.mission_status == MissionStatus.ACTIVE:
                    self._monitor_mission_progress()

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in main control loop: {e}")
                time.sleep(1.0)

    def _camera_processing_loop(self):
        """Process camera feed continuously"""
        while self.running and self.camera_active:
            try:
                ret, frame = self.camera_capture.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                # Get current altitude
                drone_state = self.mavlink_conn.get_drone_state()
                altitude = max(drone_state.altitude, 5.0)

                # Process frame with SLAM
                visual_pose = self.visual_slam.process_frame(frame, altitude)

                # Update current position
                self.current_position = LocalPoint(visual_pose.x, visual_pose.y, visual_pose.z)

                # Process obstacle avoidance
                steering, speed, emergency = self.obstacle_avoidance.process_camera_frame(
                    frame, altitude, drone_state.yaw
                )

                # Apply avoidance if needed
                if emergency or abs(steering) > 0.1 or speed < 0.8:
                    self._apply_obstacle_avoidance(steering, speed, emergency)

                # Update landmarks in path planner
                landmarks = self.visual_slam.get_landmarks()
                for lm_id, landmark in landmarks.items():
                    self.path_planner.add_visual_landmark(
                        lm_id,
                        LocalPoint(landmark.position[0], landmark.position[1], landmark.position[2]),
                        landmark.landmark_type
                    )

                # Update performance metrics
                self.performance_metrics["visual_navigation_accuracy"] = visual_pose.confidence
                self.performance_metrics["landmarks_mapped"] = len(landmarks)

                time.sleep(1.0 / self.config.get("camera_fps", 30))

            except Exception as e:
                self.logger.error(f"Camera processing error: {e}")
                time.sleep(0.1)

    def _apply_obstacle_avoidance(self, steering: float, speed: float, emergency: bool):
        """Apply obstacle avoidance commands"""
        if emergency:
            self.logger.critical("Emergency obstacle detected - stopping")
            return

        if abs(steering) > 0.1:
            # Calculate avoidance position
            drone_state = self.mavlink_conn.get_drone_state()
            avoidance_distance = 3.0

            new_x = drone_state.local_x + avoidance_distance * math.sin(drone_state.yaw + steering)
            new_y = drone_state.local_y + avoidance_distance * math.cos(drone_state.yaw + steering)

            self.drone_commands.goto_local_position(new_x, new_y, drone_state.altitude)
            self.performance_metrics["obstacles_avoided"] += 1

    def _update_system_health(self):
        """Update system health status"""
        drone_state = self.mavlink_conn.get_drone_state()

        # Update health indicators
        self.system_health["battery"] = drone_state.battery_remaining
        self.system_health["communication"] = "connected" if drone_state.timestamp > 0 else "timeout"
        self.system_health["cameras"] = "operational" if self.camera_active else "failed"

        if drone_state.visual_tracking_quality > 0.8:
            self.system_health["visual_tracking"] = "excellent"
        elif drone_state.visual_tracking_quality > 0.6:
            self.system_health["visual_tracking"] = "good"
        elif drone_state.visual_tracking_quality > 0.4:
            self.system_health["visual_tracking"] = "poor"
        else:
            self.system_health["visual_tracking"] = "lost"

        # Overall health
        if (self.system_health["battery"] < 20 or 
            self.system_health["visual_tracking"] == "lost"):
            self.system_health["overall"] = "critical"
        elif (self.system_health["battery"] < 30 or 
              self.system_health["visual_tracking"] == "poor"):
            self.system_health["overall"] = "warning"
        else:
            self.system_health["overall"] = "healthy"

    def submit_mission(self, mission: VisionMission) -> bool:
        """Submit a new vision-based mission"""
        self.logger.info(f"Submitting vision mission: {mission.mission_id}")

        # Validate mission
        if not self._validate_mission(mission):
            return False

        # Plan path
        if not self._plan_mission(mission):
            return False

        # Add to queue
        priority_value = mission.priority.value
        self.task_queue.put((priority_value, time.time(), mission))

        return True

    def _validate_mission(self, mission: VisionMission) -> bool:
        """Validate mission parameters"""
        # Check system health
        if self.system_health["overall"] == "critical":
            self.logger.error("System health critical - cannot accept mission")
            return False

        # Check battery
        if self.system_health["battery"] < 30:
            self.logger.warning("Low battery for mission")
            return False

        return True

    def _plan_mission(self, mission: VisionMission) -> bool:
        """Plan mission path"""
        # Plan path to target
        path = self.path_planner.plan_path(self.current_position, mission.target_location)

        if path is None:
            self.logger.error("No path found to target")
            return False

        self.logger.info(f"Mission path planned with {len(path)} waypoints")
        return True

    def _process_mission_queue(self):
        """Process pending missions"""
        if not self.task_queue.empty() and self.mission_status == MissionStatus.PLANNED:
            _, _, mission = self.task_queue.get()
            self.current_mission = mission
            self.start_mission()

    def start_mission(self) -> bool:
        """Start current mission"""
        if not self.current_mission:
            return False

        self.logger.info(f"Starting mission: {self.current_mission.mission_id}")

        # Navigate to target
        success = self.drone_commands.goto_local_position(
            self.current_mission.target_location.x,
            self.current_mission.target_location.y,
            self.current_mission.target_location.z
        )

        if success:
            self.mission_status = MissionStatus.ACTIVE

        return success

    def _monitor_mission_progress(self):
        """Monitor active mission progress"""
        if not self.current_mission:
            return

        # Check if target reached
        distance_to_target = self.current_position.distance_to(self.current_mission.target_location)

        if distance_to_target < 2.0:  # Within 2 meters
            self.logger.info(f"Mission {self.current_mission.mission_id} completed")
            self.mission_status = MissionStatus.COMPLETED
            self.completed_missions.append(self.current_mission)
            self.performance_metrics["missions_completed"] += 1
            self.current_mission = None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        drone_state = self.mavlink_conn.get_drone_state()

        return {
            "system_health": self.system_health,
            "mission_status": self.mission_status.value,
            "current_mission": asdict(self.current_mission) if self.current_mission else None,
            "performance_metrics": self.performance_metrics,
            "position": {
                "local_x": drone_state.local_x,
                "local_y": drone_state.local_y,
                "altitude": drone_state.altitude
            },
            "visual_navigation": {
                "tracking_quality": drone_state.visual_tracking_quality,
                "landmarks_tracked": drone_state.landmarks_tracked,
                "position_confidence": drone_state.position_confidence
            },
            "camera_status": {
                "active": self.camera_active
            },
            "discovered_landmarks": len(self.visual_slam.get_landmarks()),
            "completed_missions": len(self.completed_missions)
        }

    def stop_system(self):
        """Stop the system gracefully"""
        self.logger.info("Stopping Vision-Based Disaster Response System")

        self.running = False

        # Close camera
        if self.camera_capture:
            self.camera_capture.release()

        # Stop MAVLink connection
        if self.mavlink_conn:
            self.mavlink_conn.running = False

        self.logger.info("System stopped")

# ============================================================================
# EXAMPLE USAGE AND MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for vision-based disaster response system"""
    print("=" * 80)
    print("🎥 VISION-BASED DISASTER RESPONSE DRONE SYSTEM")
    print("=" * 80)
    print("🚁 GPS-Free Navigation using Camera-Based SLAM")
    print("🗺️  Local Coordinate System (meters from origin)")
    print("🎯 Real-time Obstacle Avoidance with Computer Vision")
    print("🏗️  Visual Landmark-Based Path Planning")
    print("=" * 80)

    # Initialize system
    disaster_system = VisionBasedDisasterSystem()

    # Start system
    if not disaster_system.start_system():
        print("❌ Failed to start system")
        return

    print("✅ Vision-based system started successfully!")
    print("\nAvailable commands:")
    print("  status    - Show system status")
    print("  mission   - Submit test mission")
    print("  takeoff   - Takeoff to 10m altitude")
    print("  land      - Land at current position")
    print("  goto X Y  - Go to local coordinates X,Y")
    print("  landmarks - Show discovered landmarks")
    print("  health    - Show system health")
    print("  quit      - Exit system")

    try:
        while True:
            command = input("\nvision_drone> ").strip().lower().split()

            if not command:
                continue

            if command[0] == "quit":
                break

            elif command[0] == "status":
                status = disaster_system.get_system_status()
                print("\n=== SYSTEM STATUS ===")
                print(f"Overall Health: {status['system_health']['overall']}")
                print(f"Mission Status: {status['mission_status']}")
                pos = status['position']
                print(f"Position: ({pos['local_x']:.1f}, {pos['local_y']:.1f}, {pos['altitude']:.1f})")
                visual = status['visual_navigation']
                print(f"Visual Tracking: {visual['tracking_quality']:.2f}")
                print(f"Landmarks: {visual['landmarks_tracked']}")
                print(f"Completed Missions: {status['completed_missions']}")

            elif command[0] == "mission":
                # Submit test mission
                test_mission = VisionMission(
                    mission_id=f"VISION_TEST_{int(time.time())}",
                    mission_type=MissionType.MEDICAL_DELIVERY,
                    priority=Priority.HIGH,
                    target_location=LocalPoint(20, 30, 15),  # 20m east, 30m north, 15m alt
                    waypoints=[LocalPoint(10, 15, 15), LocalPoint(20, 30, 15)],
                    payload_type="medical_supplies",
                    estimated_duration=300,
                    created_timestamp=time.time()
                )

                if disaster_system.submit_mission(test_mission):
                    print("✅ Vision-based mission submitted successfully")
                else:
                    print("❌ Failed to submit mission")

            elif command[0] == "takeoff":
                if disaster_system.drone_commands.takeoff(10.0):
                    print("✅ Takeoff command sent")
                else:
                    print("❌ Takeoff failed")

            elif command[0] == "goto" and len(command) >= 3:
                try:
                    x = float(command[1])
                    y = float(command[2])
                    z = 15.0  # Default altitude
                    if len(command) >= 4:
                        z = float(command[3])

                    if disaster_system.drone_commands.goto_local_position(x, y, z):
                        print(f"✅ Moving to ({x}, {y}, {z})")
                    else:
                        print("❌ Move command failed")
                except ValueError:
                    print("❌ Invalid coordinates")

            elif command[0] == "landmarks":
                landmarks = disaster_system.visual_slam.get_landmarks()
                print(f"\n=== DISCOVERED LANDMARKS ({len(landmarks)}) ===")
                for lm_id, landmark in list(landmarks.items())[:10]:  # Show first 10
                    pos = landmark.position
                    print(f"ID {lm_id}: {landmark.landmark_type} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                if len(landmarks) > 10:
                    print(f"... and {len(landmarks) - 10} more landmarks")

            elif command[0] == "health":
                health = disaster_system.system_health
                print("\n=== SYSTEM HEALTH ===")
                for component, status in health.items():
                    if component == "battery":
                        print(f"{component.title()}: {status}%")
                    else:
                        print(f"{component.title()}: {status}")

            else:
                print(f"❓ Unknown command: {command[0]}")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 Shutting down vision-based system...")
        disaster_system.stop_system()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main()
