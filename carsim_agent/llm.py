import os
import re
import requests
from typing import Dict, Any, List
from .utils.logger import log_info, log_warning, log_error

class CarsimLLMInterface:
    def __init__(self, api_endpoint: str = None):
        self.api_endpoint = api_endpoint or os.environ.get("CARSIM_LLM_ENDPOINT")
        
    def generate_mods(self, prompt: str, library_context: List[Dict]) -> Dict[str, Any]:
        if not self.api_endpoint:
            log_warning("LLM API endpoint not configured, falling back to local parsing.")
            return self._fallback_local_parse(prompt)

        payload = {
            "prompt": prompt,
            "library_context": library_context,
            "instruction": "You are a Carsim simulation expert. Convert the user's natural language requirement into a JSON object of parameter modifications. Use the provided library context to find exact parameter names. If a unit conversion is needed (e.g. km/h to m/s, or matching the library unit), handle it. Return ONLY the JSON object."
        }
        
        try:
            log_info(f"Sending request to LLM endpoint: {self.api_endpoint}")
            response = requests.post(self.api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            log_info("LLM request successful")
            return result.get("mods", result)
        except Exception as e:
            log_error(f"LLM API call failed: {e}")
            return self._fallback_local_parse(prompt)

    def _fallback_local_parse(self, prompt: str) -> Dict[str, Any]:
        mods = {}
        # 改进的正则匹配，尝试查找关键词后的数值
        speed_match = re.search(r'(?:车速|速度).*?(\d+\.?\d*)', prompt)
        if speed_match:
            val = float(speed_match.group(1))
            mods["SPEED_TARGET_CONSTANT(1)"] = val
            log_info(f"Local parse: detected speed modification -> {val}")
                
        friction_match = re.search(r'(?:摩擦|friction).*?(\d+\.?\d*)', prompt, re.IGNORECASE)
        if friction_match:
             val = float(friction_match.group(1))
             mods["MU_ROAD_CONSTANT(1)"] = val
             log_info(f"Local parse: detected friction modification -> {val}")
                 
        mods["OPT_ECHO_ALL_PARS"] = 1
        return mods
