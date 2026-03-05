import os
import re
import json
from typing import Dict, List, Optional
from .utils.logger import log_info, log_warning, log_error

class CarsimLibrary:
    def __init__(self, echo_path: str = None):
        self.library = {}
        self.section = "GENERAL"
        self.index = {}
        if echo_path and os.path.exists(echo_path):
            self.load_from_echo(echo_path)

    def load_from_echo(self, file_path: str):
        if not os.path.exists(file_path):
            log_warning(f"Echo file not found: {file_path}")
            return
        
        # 正则表达式
        bar_re = re.compile(r'^![-]{5,}$')
        cat_line_re = re.compile(r'^!\s*([A-Z][A-Z0-9 _:\-/\(\)]+)$')
        param_re = re.compile(r'^([A-Za-z0-9_()]+)\s+([^;!]+?)(?:\s*;\s*([^!]+?))?\s*(?:!\s*(.*))?$')
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            log_error(f"Failed to read echo file: {e}")
            return
            
        i = 0
        count = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # 分类检测
            if bar_re.match(line):
                # 向下查找直到找到分类标题或超出范围
                j = i + 1
                found = False
                while j < len(lines) and j <= i + 5:
                    cand = lines[j].strip()
                    m = cat_line_re.match(cand)
                    if m:
                        self.section = m.group(1).strip()
                        found = True
                        break
                    j += 1
                i = j + 1 if found else i + 1
                continue
            
            # 参数匹配
            m = param_re.match(line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                unit = (m.group(3) or "").strip()
                desc = (m.group(4) or "").strip()
                
                if self.section not in self.library:
                    self.library[self.section] = {}
                
                self.library[self.section][key] = {
                    "value": val,
                    "unit": unit,
                    "desc": desc
                }
                count += 1
            i += 1
        
        self.build_index()
        log_info(f"Loaded {count} parameters from {file_path}")

    def load_from_json(self, json_path: str):
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.library = json.load(f)
                self.build_index()
                log_info(f"Loaded parameter library from JSON: {json_path}")
            except Exception as e:
                log_error(f"Failed to load JSON library: {e}")

    def merge_from_echo(self, file_path: str):
        """增量合并，只有当参数库无该参数名时才增加"""
        temp_lib = CarsimLibrary(file_path)
        merged_count = 0
        
        # 建立当前库的所有 key 集合用于快速查重
        current_keys = set()
        for section_content in self.library.values():
            current_keys.update(section_content.keys())

        for section, params in temp_lib.library.items():
            if section not in self.library:
                self.library[section] = {}
            
            for key, data in params.items():
                if key not in current_keys:
                    self.library[section][key] = data
                    current_keys.add(key)
                    merged_count += 1
        
        self.build_index()
        log_info(f"Merged {merged_count} new parameters from {file_path}")

    def save_to_json(self, json_path: str):
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.library, f, ensure_ascii=False, indent=2)
            log_info(f"Saved parameter library to {json_path}")
        except Exception as e:
            log_error(f"Failed to save library JSON: {e}")

    def build_index(self):
        self.index = {}
        def tokens(s):
            return [t for t in re.split(r'[^A-Za-z0-9]+', s.upper()) if t]
        
        for section, params in self.library.items():
            for k in params:
                for t in tokens(k):
                    self.index.setdefault(t, set()).add((section, k))

    def get_parameter(self, key: str) -> Optional[Dict]:
        """跨 section 查找参数"""
        for section, params in self.library.items():
            if key in params:
                return {"key": key, "section": section, **params[key]}
        return None

    def get_parameter_context(self, limit: int = 500) -> List[Dict]:
        """返回用于 LLM 上下文的精简参数列表"""
        context = []
        count = 0
        for section, params in self.library.items():
            for key, item in params.items():
                if count >= limit:
                    return context
                context.append({
                    "name": key,
                    "unit": item['unit'],
                    "desc": item['desc'],
                    "category": section
                })
                count += 1
        return context
