from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import Sort
from datetime import datetime
import json
import time
from collections import defaultdict

class TrafficMonitor:
    def __init__(self, video_source, model_path='yolo11x.pt', class_file='coco.names'):
        # Initialize core components
        self.load_classes(class_file)
        self.model = YOLO(model_path)
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.cap = cv2.VideoCapture(video_source)
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize tracking variables
        self.total_count = []
        self.vehicle_positions = {}
        self.vehicle_speeds = {}
        self.vehicle_info = {}
        self.vehicle_entry_times = {}
        self.class_counts = defaultdict(int)
        self.hourly_counts = defaultdict(lambda: defaultdict(int))
        
        # Detection settings
        self.conf_threshold = 0.3
        self.lane_coordinates = self.calculate_lane_coordinates()
        
        # Initialize analytics
        self.start_time = time.time()
        self.processing_times = []
        
        # Recording setup
        self.setup_video_writer()
        
    def load_classes(self, class_file):
        with open(class_file, "r") as f:
            self.classNames = [line.strip() for line in f.readlines()]
        self.vehicle_classes = ["car", "truck", "bus", "motorbike"]
            
    def calculate_lane_coordinates(self):
        """Calculate lane coordinates based on frame size"""
        return [
            int(self.frame_width * 0.1),  # x1
            int(self.frame_height * 0.6),  # y1
            int(self.frame_width * 0.9),  # x2
            int(self.frame_height * 0.6)   # y2
        ]
    
    def setup_video_writer(self):
        """Initialize video writer for recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'traffic_recording_{timestamp}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps,
            (self.frame_width, self.frame_height)
        )
        
    def calculate_speed(self, current_pos, prev_pos):
        """Calculate vehicle speed with improved accuracy"""
        distance_pixels = math.sqrt(
            (current_pos[0] - prev_pos[0]) ** 2 +
            (current_pos[1] - prev_pos[1]) ** 2
        )
        
        # Convert pixels to meters (adjust these values based on your camera setup)
        pixels_per_meter = 30  # This needs to be calibrated for your specific setup
        distance_meters = distance_pixels / pixels_per_meter
        
        # Calculate speed (meters per second to km/h)
        speed_mps = distance_meters / (1 / self.fps)
        speed_kmh = speed_mps * 3.6
        
        # Apply smoothing and limit unrealistic values
        return min(max(speed_kmh, 0), 120)
    
    def draw_analytics(self, frame):
        """Draw analytics overlay on frame"""
        # Background for analytics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display analytics
        y_offset = 40
        cv2.putText(frame, f"Total Vehicles: {len(self.total_count)}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for vehicle_class in self.vehicle_classes:
            y_offset += 30
            count = self.class_counts[vehicle_class]
            cv2.putText(frame, f"{vehicle_class.title()}: {count}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add average speed
        if self.vehicle_speeds:
            avg_speed = sum(self.vehicle_speeds.values()) / len(self.vehicle_speeds)
            y_offset += 30
            cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def save_analytics(self):
        """Save analytics data to JSON file"""
        analytics_data = {
            'total_count': len(self.total_count),
            'class_distribution': dict(self.class_counts),
            'hourly_counts': dict(self.hourly_counts),
            'average_speed': sum(self.vehicle_speeds.values()) / len(self.vehicle_speeds) if self.vehicle_speeds else 0,
            'processing_stats': {
                'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                'total_runtime': time.time() - self.start_time
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'traffic_analytics_{timestamp}.json', 'w') as f:
            json.dump(analytics_data, f, indent=4)
    
    def process_frame(self, frame):
        """Process a single frame"""
        frame_start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, stream=True)
        
        # Process detections
        detections = np.empty((0, 5))
        current_frame_detections = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                
                if (self.classNames[cls] in self.vehicle_classes and 
                    conf > self.conf_threshold):
                    width, height = x2 - x1, y2 - y1
                    if width > 30 and height > 30:
                        detection = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, detection))
                        current_frame_detections[tuple(detection)] = self.classNames[cls]
        
        # Update tracking
        tracked_objects = self.tracker.update(detections)
        
        # Draw detection line
        cv2.line(frame, 
                 (self.lane_coordinates[0], self.lane_coordinates[1]),
                 (self.lane_coordinates[2], self.lane_coordinates[3]), 
                 (0, 0, 255), 5)
        
        # Process tracked objects
        current_hour = datetime.now().strftime("%H")
        
        for tracked_obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, tracked_obj[:5])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Update vehicle classification
            if obj_id not in self.vehicle_info:
                # Find closest detection
                closest_detection = min(
                    current_frame_detections.items(),
                    key=lambda x: np.linalg.norm(np.array([cx, cy]) - 
                                               np.array([(x[0][0] + x[0][2])/2, 
                                                       (x[0][1] + x[0][3])/2])),
                    default=(None, "Unknown")
                )
                self.vehicle_info[obj_id] = closest_detection[1]
                
            # Update speed calculation
            if obj_id in self.vehicle_positions:
                current_pos = (cx, cy)
                prev_pos = self.vehicle_positions[obj_id]
                self.vehicle_speeds[obj_id] = self.calculate_speed(current_pos, prev_pos)
            
            self.vehicle_positions[obj_id] = (cx, cy)
            
            # Draw vehicle info
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, 
                            colorR=(255, 0, 255))
            cvzone.putTextRect(frame, f"{self.vehicle_info[obj_id]}", 
                             (x1, max(35, y1)), scale=1, thickness=2, 
                             colorR=(0, 255, 0))
            
            # Display speed
            speed_text = f"Speed: {int(self.vehicle_speeds.get(obj_id, 0))} km/h"
            cvzone.putTextRect(frame, speed_text, (x1, y1 - 20), scale=1, 
                             thickness=2, colorR=(0, 255, 0))
            
            # Check line crossing
            if (self.lane_coordinates[0] < cx < self.lane_coordinates[2] and 
                abs(cy - self.lane_coordinates[1]) < 15):
                if obj_id not in self.total_count:
                    self.total_count.append(obj_id)
                    self.class_counts[self.vehicle_info[obj_id]] += 1
                    self.hourly_counts[current_hour][self.vehicle_info[obj_id]] += 1
        
        # Draw analytics
        self.draw_analytics(frame)
        
        # Calculate processing time
        self.processing_times.append(time.time() - frame_start_time)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Save frame to video
                self.video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow("Traffic Monitoring", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Cleanup
            self.save_analytics()
            self.cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    monitor = TrafficMonitor(
        video_source="video3.mp4",
        model_path="yolov8n.pt",
        class_file="coco.names"
    )
    monitor.run()