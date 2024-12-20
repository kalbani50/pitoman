"""
نظام تحسين التكلفة والموارد
"""

import psutil
import torch
from typing import Dict, List
import logging
from datetime import datetime

class CostOptimizationManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
        self.power_manager = PowerManager()
        
    def optimize_resources(self):
        """تحسين استخدام الموارد"""
        self.resource_monitor.monitor()
        self.power_manager.optimize()
        self._adjust_trading_frequency()
        
    def _adjust_trading_frequency(self):
        """تعديل تردد التداول حسب النشاط"""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 80 or memory_usage > 85:
            self.config['trading_frequency'] = 'low'
        else:
            self.config['trading_frequency'] = 'normal'

class ResourceMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            'cpu': 80,
            'memory': 85,
            'gpu': 90,
            'network': 1000  # MB/s
        }
        
    def monitor(self):
        """مراقبة استخدام الموارد"""
        self.metrics['cpu'] = psutil.cpu_percent()
        self.metrics['memory'] = psutil.virtual_memory().percent
        self.metrics['disk'] = psutil.disk_usage('/').percent
        self.metrics['network'] = self._get_network_usage()
        
        if torch.cuda.is_available():
            self.metrics['gpu'] = self._get_gpu_usage()
            
    def _get_network_usage(self) -> float:
        """قياس استخدام الشبكة"""
        net_io = psutil.net_io_counters()
        return (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024

class PowerManager:
    def __init__(self):
        self.power_mode = 'balanced'
        self.gpu_enabled = torch.cuda.is_available()
        
    def optimize(self):
        """تحسين استهلاك الطاقة"""
        if self.gpu_enabled:
            self._optimize_gpu_power()
        self._optimize_cpu_power()
        
    def _optimize_gpu_power(self):
        """تحسين استهلاك GPU"""
        if not self._is_gpu_needed():
            torch.cuda.empty_cache()
            # تعطيل GPU عند عدم الحاجة
            
    def _optimize_cpu_power(self):
        """تحسين استهلاك CPU"""
        if self.power_mode == 'eco':
            # تقليل تردد المعالج
            pass

class NetworkOptimizer:
    def __init__(self):
        self.connections = {}
        self.bandwidth_usage = {}
        
    def optimize_connections(self):
        """تحسين اتصالات الشبكة"""
        self._optimize_websocket_connections()
        self._optimize_rest_connections()
        self._implement_data_compression()
        
    def _optimize_websocket_connections(self):
        """تحسين اتصالات WebSocket"""
        # الاحتفاظ فقط بالاتصالات الضرورية لـ Binance و OKX
        active_connections = ['binance_futures', 'okx_spot', 'okx_futures']
        for conn in self.connections:
            if conn not in active_connections:
                self.connections[conn].close()
                
    def _implement_data_compression(self):
        """تطبيق ضغط البيانات"""
        # تطبيق ضغط GZIP على طلبات HTTP
        # تقليل حجم البيانات المرسلة والمستلمة
        pass

class DataOptimizer:
    def __init__(self):
        self.data_retention = {}
        self.cache_settings = {}
        
    def optimize_storage(self):
        """تحسين التخزين"""
        self._cleanup_old_data()
        self._optimize_cache()
        self._compress_historical_data()
        
    def _cleanup_old_data(self):
        """تنظيف البيانات القديمة"""
        current_time = datetime.now()
        for key, data in self.data_retention.items():
            if (current_time - data['timestamp']).days > 30:
                del self.data_retention[key]
                
    def _optimize_cache(self):
        """تحسين التخزين المؤقت"""
        # الاحتفاظ فقط ببيانات Binance و OKX
        allowed_exchanges = ['binance', 'okx']
        for key in list(self.cache_settings.keys()):
            if not any(exchange in key for exchange in allowed_exchanges):
                del self.cache_settings[key]
