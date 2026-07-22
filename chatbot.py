import os
import json
import requests
from threading import Thread

class Chatbot:
    def __init__(self, api_key=None):
        """
        Hybrid Chatbot Interface with 3-second LLM timeout and rule-based fallback.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.state = {
            "left_vehicle_count": 0,
            "right_vehicle_count": 0,
            "current_signal_state": "GREEN_LEFT",
            "remaining_signal_time": 10.0,
            "average_speed_left": 0.0,
            "average_speed_right": 0.0,
            "total_vehicles_processed": 0
        }

    def update_state(self, left_count, right_count, signal_state, remaining_time, avg_speed_l, avg_speed_r, total_vehicles):
        """
        Updates the shared traffic state.
        """
        self.state["left_vehicle_count"] = left_count
        self.state["right_vehicle_count"] = right_count
        self.state["current_signal_state"] = signal_state
        self.state["remaining_signal_time"] = remaining_time
        self.state["average_speed_left"] = avg_speed_l
        self.state["average_speed_right"] = avg_speed_r
        self.state["total_vehicles_processed"] = total_vehicles

    def query(self, user_query):
        """
        Main query interface: tries AI tier first, falls back to rule-based tier.
        """
        if self.api_key:
            try:
                # Run the API call with timeout
                result = []
                def call_api():
                    try:
                        ans = self._call_gemini_api(user_query)
                        result.append(ans)
                    except Exception as e:
                        result.append(None)
                
                t = Thread(target=call_api)
                t.start()
                t.join(timeout=3.0)  # 3-second timeout
                
                if t.is_alive():
                    print("[Chatbot] LLM API timeout (3s limit reached). Falling back to rule-based tier.")
                elif result and result[0] is not None:
                    return f"[AI Tier] {result[0]}"
            except Exception as e:
                print(f"[Chatbot] LLM API error: {e}. Falling back to rule-based tier.")
        
        # Fallback to rule-based tier
        return f"[Fallback Tier] {self._rule_based_response(user_query)}"

    def _call_gemini_api(self, query_text):
        """
        Makes a direct POST call to the Gemini API.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        prompt = (
            f"You are an AI assistant built into an Intelligent Traffic Analyzer system.\n"
            f"Here is the current traffic state in JSON format:\n"
            f"{json.dumps(self.state, indent=2)}\n\n"
            f"The user is asking: '{query_text}'\n"
            f"Answer the user's query concisely based on the current traffic state. Keep it under 2 sentences."
        )
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=3.0)
        if response.status_code == 200:
            res_data = response.json()
            try:
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            except KeyError:
                return "Error parsing LLM response."
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def _rule_based_response(self, query_text):
        """
        Classifies intent and generates a templated response from the shared traffic state.
        """
        q = query_text.lower()
        
        # Intent classification keywords
        count_keywords = ["count", "how many", "number", "cars", "vehicles", "trucks", "buses", "motorcycles", "total"]
        signal_keywords = ["signal", "light", "phase", "green", "red", "color", "state"]
        speed_keywords = ["speed", "fast", "velocity", "km/h", "kph", "average speed", "speeds"]
        density_keywords = ["density", "busy", "busier", "congested", "congestion", "crowded", "lane"]
        status_keywords = ["status", "report", "summary", "general", "overall"]

        left = self.state["left_vehicle_count"]
        right = self.state["right_vehicle_count"]
        total = left + right
        sig = self.state["current_signal_state"]
        timer = self.state["remaining_signal_time"]
        spd_l = self.state["average_speed_left"]
        spd_r = self.state["average_speed_right"]
        tot_proc = self.state["total_vehicles_processed"]

        # 1. Vehicle Count Intent
        if any(w in q for w in count_keywords):
            return (f"Currently, there are {left} vehicles in the Left lane and {right} vehicles in the Right lane, "
                    f"totaling {total} active vehicles on screen. A total of {tot_proc} vehicles have been processed.")
        
        # 2. Signal Status Intent
        if any(w in q for w in signal_keywords):
            return f"The traffic signal is currently {sig} with {timer:.1f} seconds remaining in this phase."
        
        # 3. Speed Intent
        if any(w in q for w in speed_keywords):
            return (f"The average speed in the Left lane is {spd_l:.1f} km/h, "
                    f"and in the Right lane it is {spd_r:.1f} km/h.")
        
        # 4. Density Intent
        if any(w in q for w in density_keywords):
            if left > right:
                busy = "Left"
            elif right > left:
                busy = "Right"
            else:
                busy = "Neither"
            return (f"Lane densities are Left: {left} and Right: {right}. "
                    f"{busy + ' lane' if busy != 'Neither' else 'Both lanes'} are currently busier.")
        
        # 5. General Status Intent
        if any(w in q for w in status_keywords):
            return (f"Traffic Status Report: Signal is {sig} ({timer:.1f}s left). "
                    f"Left lane has {left} vehicles (avg speed {spd_l:.1f} km/h). "
                    f"Right lane has {right} vehicles (avg speed {spd_r:.1f} km/h).")

        # 6. Unrecognized Fallback
        return ("I'm sorry, I couldn't recognize your query. You can ask me about vehicle counts, "
                "signal status, speeds, density, or general status.")
