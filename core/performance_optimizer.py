"""
نظام تحسين الأداء والموارد
"""

import psutil
import torch
import numpy as np
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile

class PerformanceOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_threads', 8))
        self.process_pool = ProcessPoolExecutor(max_workers=config.get('max_processes', 4))
        self.gpu_enabled = torch.cuda.is_available()
        
    def optimize_memory(self):
        """تحسين استخدام الذاكرة"""
        torch.cuda.empty_cache()  # تنظيف ذاكرة GPU
        gc.collect()  # تنظيف الذاكرة العادية
        
    @profile
    def process_data_batch(self, data: List):
        """معالجة البيانات بكفاءة"""
        if self.gpu_enabled:
            return self._process_gpu(data)
        return self._process_cpu(data)
        
    def _process_gpu(self, data: List):
        """معالجة على GPU"""
        with torch.cuda.device(0):
            tensor_data = torch.tensor(data)
            return self._parallel_process(tensor_data)
            
    def _process_cpu(self, data: List):
        """معالجة على CPU"""
        return self.process_pool.map(self._process_single, data)
        
    def _parallel_process(self, data):
        """معالجة متوازية"""
        chunk_size = len(data) // psutil.cpu_count()
        chunks = np.array_split(data, chunk_size)
        return self.thread_pool.map(self._process_chunk, chunks)

class LoadBalancer:
    def __init__(self):
        self.servers = []
        self.current_index = 0
        
    def add_server(self, server):
        self.servers.append(server)
        
    def get_next_server(self):
        """توزيع الحمل بشكل دائري"""
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server

class CacheOptimizer:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        
    def get(self, key):
        """الحصول على قيمة مع تحديث عداد الوصول"""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
        
    def put(self, key, value):
        """إضافة قيمة مع التحكم في حجم الذاكرة المؤقتة"""
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        self.cache[key] = value
        self.access_count[key] = 1
        
    def _evict_least_used(self):
        """إزالة العناصر الأقل استخداماً"""
        min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        del self.cache[min_key]
        del self.access_count[min_key]
