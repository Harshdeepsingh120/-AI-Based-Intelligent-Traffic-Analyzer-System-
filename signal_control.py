import time

class SignalController:
    def __init__(self, fps=30.0, t_eval=5.0, t_min=10.0, t_max=60.0, t_all_red=2.0):
        """
        Adaptive Traffic Signal Control State Machine.
        """
        self.fps = fps
        self.t_eval = t_eval            # Re-check every 5 seconds
        self.t_min = t_min              # Min green phase: 10s
        self.t_max = t_max              # Max green phase: 60s
        self.t_all_red = t_all_red      # All-red gap: 2s
        
        self.state = "GREEN_LEFT"       # GREEN_LEFT, GREEN_RIGHT, ALL_RED
        self.timer = 10.0               # Remaining time in seconds for current state
        self.elapsed_in_state = 0.0     # Time elapsed in current state
        self.target_state = None        # Next green state after ALL_RED
        
        # Log of transitions for latency / responsiveness measurement
        self.transition_log = []
        
        # Keep track of winner density for log
        self.last_winner_density = 0
        self.last_loser_density = 0

    def get_t_green(self, d_winner, d_loser):
        """
        Calculates T_green using the density formula.
        T_green = T_min + (T_max - T_min) * (D_winner / (D_winner + D_loser))
        """
        if d_winner + d_loser == 0:
            return self.t_min
        ratio = d_winner / float(d_winner + d_loser)
        return self.t_min + (self.t_max - self.t_min) * ratio

    def update(self, d_l, d_r, frame_number):
        """
        Updates the state machine by one frame.
        """
        dt = 1.0 / self.fps
        self.timer -= dt
        self.elapsed_in_state += dt

        # State transitions
        if self.state == "ALL_RED":
            if self.timer <= 0:
                old_state = self.state
                self.state = self.target_state
                d_winner = max(d_l, d_r)
                d_loser = min(d_l, d_r)
                t_green = self.get_t_green(d_winner, d_loser)
                self.timer = t_green
                self.elapsed_in_state = 0.0
                self.target_state = None
                
                event = {
                    "frame": frame_number,
                    "timestamp": time.time(),
                    "from_state": old_state,
                    "to_state": self.state,
                    "duration": t_green,
                    "density_left": d_l,
                    "density_right": d_r,
                    "reason": "ALL_RED phase expired"
                }
                self.transition_log.append(event)
                print(f"[Signal] Transition: ALL_RED -> {self.state} for {t_green:.1f}s (D_l={d_l}, D_r={d_r})")

        elif self.state in ["GREEN_LEFT", "GREEN_RIGHT"]:
            # 5-second periodic evaluation trigger
            eval_frames = int(self.t_eval * self.fps)
            if frame_number % eval_frames == 0:
                # Check if a switch is warranted
                if self.state == "GREEN_LEFT" and d_r > d_l:
                    if self.elapsed_in_state >= self.t_min or self.timer <= 0:
                        self.initiate_transition("GREEN_RIGHT", d_l, d_r, frame_number, "Right lane busier (5s check)")
                elif self.state == "GREEN_RIGHT" and d_l > d_r:
                    if self.elapsed_in_state >= self.t_min or self.timer <= 0:
                        self.initiate_transition("GREEN_LEFT", d_l, d_r, frame_number, "Left lane busier (5s check)")

            # Force switch if max timer expires
            if self.timer <= 0:
                if self.state == "GREEN_LEFT" and d_r > 0:
                    self.initiate_transition("GREEN_RIGHT", d_l, d_r, frame_number, "Max green expired & vehicles in Right")
                elif self.state == "GREEN_RIGHT" and d_l > 0:
                    self.initiate_transition("GREEN_LEFT", d_l, d_r, frame_number, "Max green expired & vehicles in Left")
                else:
                    # No vehicles in opposite lane, extend current green by T_min
                    self.timer = self.t_min
                    print(f"[Signal] Green extended for {self.state} by {self.t_min:.1f}s (opposite lane empty)")

    def initiate_transition(self, target_state, d_l, d_r, frame_number, reason):
        """
        Starts the transition to ALL_RED.
        """
        old_state = self.state
        self.state = "ALL_RED"
        self.timer = self.t_all_red
        self.elapsed_in_state = 0.0
        self.target_state = target_state
        
        event = {
            "frame": frame_number,
            "timestamp": time.time(),
            "from_state": old_state,
            "to_state": "ALL_RED",
            "duration": self.t_all_red,
            "density_left": d_l,
            "density_right": d_r,
            "reason": reason
        }
        self.transition_log.append(event)
        print(f"[Signal] Transition: {old_state} -> ALL_RED for {self.t_all_red:.1f}s. Target: {target_state} (Reason: {reason})")

    def get_status(self):
        """
        Returns signal status dict.
        """
        return {
            "state": self.state,
            "timer": max(0.0, self.timer),
            "elapsed": self.elapsed_in_state
        }
