# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np

def create_sample_delivery(output_path="cricket.mp4", num_frames=60, width=1280, height=720, fps=25):
    """
    Generates a realistic broadcast-style cricket clip for demonstration and testing.
    Draws pitch, crease, stumps, and a bowling trajectory.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Ground color (deep cricket grass green)
    ground_color = (35, 120, 45)
    # Pitch color (pale dry clay/grass)
    pitch_color = (130, 185, 195)
    
    # Stump region coordinates at the striker's end (right side of camera view)
    stump_x1, stump_y1 = 920, 290
    stump_x2, stump_y2 = 960, 460
    
    # Trajectory parameters for the ball:
    # Starts bowler end (left): (200, 360)
    # Bounces around frame 30: (620, 480)
    # Reaches stumps around frame 55: (940, 390)
    
    for i in range(num_frames):
        frame = np.full((height, width, 3), ground_color, dtype=np.uint8)
        
        # Draw perspective pitch
        pitch_pts = np.array([
            [100, 490],   # bowler crease bottom-left
            [1100, 490],  # striker crease bottom-right
            [1050, 430],  # striker top-right
            [180, 430]    # bowler top-left
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pitch_pts], pitch_color)
        
        # Crease markings (white lines)
        cv2.line(frame, (250, 430), (220, 490), (255, 255, 255), 3) # bowling crease
        cv2.line(frame, (900, 430), (920, 490), (255, 255, 255), 3) # popping crease
        
        # Draw striker's Stumps & Bails
        # 3 stumps
        stump_color = (220, 240, 245)
        for offset in [0, 15, 30]:
            cv2.line(frame, (stump_x1 + offset, stump_y2), (stump_x1 + offset, stump_y1), stump_color, 4)
        # Bails
        cv2.line(frame, (stump_x1 - 4, stump_y1), (stump_x1 + 34, stump_y1), (200, 220, 230), 4)
        
        # Calculate ball position (projectile with bounce)
        t = i / float(num_frames)
        if i <= 32:
            # Flight 1: Bowler release to pitch bounce
            prog = i / 32.0
            bx = int(180 + (640 - 180) * prog)
            # Parabolic descent: starts at 280, hits pitch at 470
            by = int(280 + 190 * (prog ** 1.3))
            ball_radius = int(8 + 8 * prog) # Ball gets larger as it approaches camera
        else:
            # Flight 2: Post-bounce trajectory towards stumps
            prog = (i - 32) / (num_frames - 32.0)
            bx = int(640 + (1020 - 640) * prog)
            # Rebound arc heading directly into the stumps at y ~ 370
            by = int(470 - 130 * (prog ** 0.8))
            ball_radius = int(16 + 6 * prog)
            
        # Draw cricket ball (classic red leather with white seam highlight)
        cv2.circle(frame, (bx, by), ball_radius, (25, 25, 200), -1) # Red leather
        cv2.circle(frame, (bx, by), ball_radius, (10, 10, 130), 2)  # Outer rim
        cv2.ellipse(frame, (bx, by), (ball_radius, ball_radius // 2), 45, 0, 360, (230, 230, 230), 1) # Seam
        
        # Add a subtle delivery information banner on top
        cv2.putText(frame, "SIMULATED BROADCAST FEED - LBW ANGLE", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        
    out.release()
    print(f"Sample cricket video generated successfully: {output_path} ({num_frames} frames, {width}x{height} @ {fps}fps)")

if __name__ == "__main__":
    create_sample_delivery()
