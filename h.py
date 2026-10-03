"""
Antigravity Hand Tracking - Python Version
Mirrors the HTML/JavaScript functionality using OpenCV and MediaPipe
"""

import os
# Suppress MediaPipe C++ logging
os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FXConfig:
    glow: bool = True
    trails: bool = True
    skeleton: bool = True
    particles: bool = True

class Colors:
    GLOW = (175, 169, 236)      # Purple (BGR)
    TRAIL = (183, 74, 83)       # Deep purple (BGR)
    SKELETON = (165, 202, 93)   # Teal (BGR)
    JOINT = (254, 237, 238)     # Light white (BGR)
    WRIST = (236, 169, 175)     # Purple-pink (BGR)
    PARTICLE = (117, 199, 250)  # Amber (BGR)

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: float

class ParticleSystem:
    def __init__(self):
        self.particles: List[Particle] = []
    
    def spawn(self, x: float, y: float, count: int = 1):
        for _ in range(count):
            self.particles.append(Particle(
                x=x,
                y=y,
                vx=(random.random() - 0.5) * 3.5,
                vy=-random.random() * 4 - 1,
                life=1.0,
                size=random.random() * 3.5 + 1
            ))
    
    def update_and_draw(self, canvas: np.ndarray) -> np.ndarray:
        height, width = canvas.shape[:2]
        
        # Update in reverse to allow removal
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.06  # gravity
            p.life -= 0.022
            
            if p.life <= 0:
                self.particles.pop(i)
                continue
            
            # Draw particle
            if 0 <= int(p.x) < width and 0 <= int(p.y) < height:
                alpha = p.life * 0.85
                radius = int(p.size * p.life)
                color = tuple(int(c * alpha) for c in Colors.PARTICLE)
                cv2.circle(canvas, (int(p.x), int(p.y)), max(1, radius), color, -1)
        
        return canvas

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class TrailSystem:
    def __init__(self, max_length: int = 55):
        self.max_length = max_length
        self.trails: List[List[Tuple[float, float]]] = [[], []]  # Left, Right hands
    
    def add_point(self, hand_index: int, x: float, y: float):
        if hand_index < len(self.trails):
            self.trails[hand_index].append((x, y))
            if len(self.trails[hand_index]) > self.max_length:
                self.trails[hand_index].pop(0)
    
    def draw(self, canvas: np.ndarray, hand_index: int):
        if hand_index >= len(self.trails):
            return canvas
        
        trail = self.trails[hand_index]
        if len(trail) < 2:
            return canvas
        
        for i in range(1, len(trail)):
            t = i / len(trail)
            alpha = t * 0.65
            thickness = max(1, int(t * 5))
            
            color = tuple(int(c * alpha) for c in Colors.TRAIL)
            cv2.line(canvas, 
                    (int(trail[i-1][0]), int(trail[i-1][1])),
                    (int(trail[i][0]), int(trail[i][1])),
                    color, thickness, cv2.LINE_AA)
        
        return canvas

# ═══════════════════════════════════════════════════════════════════════════════
# HAND RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class HandRenderer:
    # MediaPipe hand connections (skeleton)
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),# ring
        (0, 17), (17, 18), (18, 19), (19, 20),# pinky
        (5, 9), (9, 13), (13, 17),            # palm arc
    ]
    
    FINGERTIPS = [4, 8, 12, 16, 20]
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.trail_system = TrailSystem()
        self.particle_system = ParticleSystem()
    
    def draw_glow(self, canvas: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw glow effect around palm and fingertips"""
        palm_x, palm_y = landmarks[0]
        
        # Palm glow
        overlay = canvas.copy()
        cv2.circle(overlay, (palm_x, palm_y), 110, (*Colors.GLOW, 56), -1)
        cv2.addWeighted(canvas, 1, overlay, 0.22, 0, canvas)
        
        # Fingertip glows
        for tip_idx in self.FINGERTIPS:
            tip_x, tip_y = landmarks[tip_idx]
            overlay = canvas.copy()
            cv2.circle(overlay, (tip_x, tip_y), 28, (*Colors.GLOW, 71), -1)
            cv2.addWeighted(canvas, 1, overlay, 0.28, 0, canvas)
        
        return canvas
    
    def draw_skeleton(self, canvas: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw hand skeleton connections and joints"""
        # Draw connections
        for a, b in self.CONNECTIONS:
            pt1 = landmarks[a]
            pt2 = landmarks[b]
            color = tuple(int(c * 0.65) for c in Colors.SKELETON)
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        
        # Draw joints
        for i, (x, y) in enumerate(landmarks):
            is_wrist = i == 0
            is_tip = i in self.FINGERTIPS
            radius = 6 if is_wrist else (5 if is_tip else 3)
            color = Colors.WRIST if is_wrist else Colors.JOINT
            
            # Joint glow
            overlay = canvas.copy()
            cv2.circle(overlay, (x, y), radius + 5, (*Colors.GLOW, 30), -1)
            cv2.addWeighted(canvas, 1, overlay, 0.12, 0, canvas)
            
            # Joint core
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        
        return canvas
    
    def draw_hand(self, canvas: np.ndarray, landmarks: List, hand_index: int, 
                  fx: FXConfig) -> np.ndarray:
        """Draw a single hand with all effects"""
        # Convert normalized coordinates to pixels (mirror x)
        pts = []
        # Support both legacy API (.landmark) and new Tasks API (direct list)
        for lm in getattr(landmarks, 'landmark', landmarks):
            x = int((1 - lm.x) * self.width)
            y = int(lm.y * self.height)
            pts.append((x, y))
        
        palm_x, palm_y = pts[0]
        
        # Glow effect
        if fx.glow:
            canvas = self.draw_glow(canvas, pts)
        
        # Trails
        if fx.trails:
            self.trail_system.add_point(hand_index, palm_x, palm_y)
            canvas = self.trail_system.draw(canvas, hand_index)
        
        # Skeleton
        if fx.skeleton:
            canvas = self.draw_skeleton(canvas, pts)
        
        # Particles
        if fx.particles:
            for tip_idx in self.FINGERTIPS:
                if random.random() < 0.4:
                    tip_x, tip_y = pts[tip_idx]
                    self.particle_system.spawn(tip_x, tip_y)
        
        return canvas
    
    def update_particles(self, canvas: np.ndarray) -> np.ndarray:
        """Update and draw all particles"""
        return self.particle_system.update_and_draw(canvas)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class AntigravityHandTracking:
    def __init__(self):
        self.fx = FXConfig()
        self.cap = None
        self.renderer = None
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.vision = vision
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6)
        self.detector = None
        self.running = False
        
        # UI state
        self.show_help = True
        
    def initialize(self):
        """Initialize camera and MediaPipe"""
        print("🚀 Initializing Antigravity Hand Tracking...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual dimensions
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Camera resolution: {width}x{height}")
        
        # Initialize renderer
        self.renderer = HandRenderer(width, height)
        
        # Initialize MediaPipe Hands using Tasks API
        self.detector = self.vision.HandLandmarker.create_from_options(self.options)
        
        self.running = True
        print("✅ Ready! Press 'Q' to quit, 'H' for help")
        
    def draw_hud(self, canvas: np.ndarray, num_hands: int) -> np.ndarray:
        """Draw heads-up display"""
        h, w = canvas.shape[:2]
        
        # Dark background for text area
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (7, 7, 15), -1)
        cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0, canvas)
        
        # Title
        cv2.putText(canvas, "ANTIGRAVITY HAND TRACKING", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 169, 236), 1, cv2.LINE_AA)
        
        # Status
        status = f"Hands detected: {num_hands}" if num_hands > 0 else "Show your hands"
        status_color = (93, 202, 165) if num_hands > 0 else (175, 169, 236)
        cv2.putText(canvas, status, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
        
        # FX status
        fx_text = f"[1]Glow:{'ON' if self.fx.glow else 'OFF'}  [2]Trails:{'ON' if self.fx.trails else 'OFF'}  [3]Skeleton:{'ON' if self.fx.skeleton else 'OFF'}  [4]Particles:{'ON' if self.fx.particles else 'OFF'}"
        cv2.putText(canvas, fx_text, (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (175, 169, 236), 1, cv2.LINE_AA)
        
        # Controls hint
        cv2.putText(canvas, "Controls: 1-4=toggle effects, R=reset trails, H=help, Q=quit", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)
        
        # Help overlay
        if self.show_help:
            help_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            help_overlay[:] = (7, 7, 15)
            
            help_texts = [
                "CONTROLS",
                "",
                "1 - Toggle Glow Effect",
                "2 - Toggle Trails Effect", 
                "3 - Toggle Skeleton Effect",
                "4 - Toggle Particles Effect",
                "R - Reset Trail History",
                "H - Toggle This Help",
                "Q - Quit Application",
                "",
                "Press any key to continue..."
            ]
            
            y_start = h // 2 - len(help_texts) * 15
            for i, text in enumerate(help_texts):
                y = y_start + i * 30
                if text == "CONTROLS":
                    cv2.putText(help_overlay, text, (w//2 - 80, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (175, 169, 236), 2, cv2.LINE_AA)
                elif text.startswith("Press"):
                    cv2.putText(help_overlay, text, (w//2 - 120, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
                else:
                    cv2.putText(help_overlay, text, (w//2 - 150, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.addWeighted(canvas, 0.3, help_overlay, 0.7, 0, canvas)
        
        return canvas
    
    def handle_input(self, key: int):
        """Handle keyboard input"""
        if key == ord('q') or key == ord('Q'):
            self.running = False
        elif key == ord('1'):
            self.fx.glow = not self.fx.glow
            print(f"Glow: {'ON' if self.fx.glow else 'OFF'}")
        elif key == ord('2'):
            self.fx.trails = not self.fx.trails
            print(f"Trails: {'ON' if self.fx.trails else 'OFF'}")
        elif key == ord('3'):
            self.fx.skeleton = not self.fx.skeleton
            print(f"Skeleton: {'ON' if self.fx.skeleton else 'OFF'}")
        elif key == ord('4'):
            self.fx.particles = not self.fx.particles
            print(f"Particles: {'ON' if self.fx.particles else 'OFF'}")
        elif key == ord('r') or key == ord('R'):
            self.renderer.trail_system.trails = [[], []]
            print("Trails reset")
        elif key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
    
    def run(self):
        """Main loop"""
        try:
            self.initialize()
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Failed to capture frame")
                    continue
                
                # Mirror frame horizontally (like the web version)
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hands with Tasks API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.detector.detect(mp_image)
                
                # Create canvas for effects (dark background)
                canvas = np.zeros_like(frame)
                canvas[:] = (15, 7, 7)  # Dark purple-ish background
                
                # Blend video feed (subtle)
                video_overlay = cv2.addWeighted(frame, 0.45, canvas, 0.55, 0)
                
                num_hands = 0
                if hasattr(results, 'hand_landmarks') and results.hand_landmarks:
                    num_hands = len(results.hand_landmarks)
                    for i, landmarks in enumerate(results.hand_landmarks):
                        video_overlay = self.renderer.draw_hand(
                            video_overlay, landmarks, i, self.fx
                        )
                
                # Update particles (always update even if no hands for fading out)
                if self.fx.particles:
                    video_overlay = self.renderer.update_particles(video_overlay)
                
                # Draw HUD
                video_overlay = self.draw_hud(video_overlay, num_hands)
                
                # Display
                cv2.imshow('Antigravity Hand Tracking', video_overlay)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_input(key)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = AntigravityHandTracking()
    app.run()
