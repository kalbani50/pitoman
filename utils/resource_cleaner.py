import os
import shutil
import logging
import time
from datetime import datetime, timedelta
import psutil
import gc
from typing import List, Dict
import json

class ResourceCleaner:
    """نظام تنظيف وإدارة الموارد للبوت"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'max_log_age_days': 7,
            'max_cache_age_hours': 24,
            'max_temp_age_hours': 12,
            'memory_threshold': 85,  # نسبة استخدام الذاكرة التي تستدعي التنظيف
            'disk_threshold': 90,    # نسبة استخدام القرص التي تستدعي التنظيف
            'paths': {
                'logs': 'logs',
                'cache': 'data/cache',
                'temp': 'data/temp',
                'historical': 'data/historical'
            }
        }
        
    def clean_old_logs(self) -> int:
        """تنظيف ملفات السجلات القديمة"""
        cleaned = 0
        log_dir = self.config['paths']['logs']
        if not os.path.exists(log_dir):
            return cleaned
            
        cutoff = datetime.now() - timedelta(days=self.config['max_log_age_days'])
        
        for root, _, files in os.walk(log_dir):
            for file in files:
                if file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                        try:
                            os.remove(file_path)
                            cleaned += 1
                            self.logger.info(f"تم حذف ملف السجل القديم: {file}")
                        except Exception as e:
                            self.logger.error(f"خطأ في حذف الملف {file}: {str(e)}")
        
        return cleaned

    def clean_cache(self) -> int:
        """تنظيف ملفات الكاش المؤقتة"""
        cleaned = 0
        cache_dir = self.config['paths']['cache']
        if not os.path.exists(cache_dir):
            return cleaned
            
        cutoff = datetime.now() - timedelta(hours=self.config['max_cache_age_hours'])
        
        for root, _, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                    try:
                        os.remove(file_path)
                        cleaned += 1
                        self.logger.info(f"تم حذف ملف الكاش: {file}")
                    except Exception as e:
                        self.logger.error(f"خطأ في حذف الكاش {file}: {str(e)}")
        
        return cleaned

    def clean_temp_files(self) -> int:
        """تنظيف الملفات المؤقتة"""
        cleaned = 0
        temp_dir = self.config['paths']['temp']
        if not os.path.exists(temp_dir):
            return cleaned
            
        cutoff = datetime.now() - timedelta(hours=self.config['max_temp_age_hours'])
        
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                    try:
                        os.remove(file_path)
                        cleaned += 1
                        self.logger.info(f"تم حذف الملف المؤقت: {file}")
                    except Exception as e:
                        self.logger.error(f"خطأ في حذف الملف المؤقت {file}: {str(e)}")
        
        return cleaned

    def optimize_memory(self) -> Dict:
        """تحسين استخدام الذاكرة"""
        before_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # تنظيف الذاكرة غير المستخدمة
        gc.collect()
        
        # تنظيف ذاكرة التخزين المؤقت للنظام
        if psutil.virtual_memory().percent > self.config['memory_threshold']:
            self.clean_cache()
        
        after_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        saved = before_mem - after_mem
        
        return {
            'before_mb': round(before_mem, 2),
            'after_mb': round(after_mem, 2),
            'saved_mb': round(saved, 2)
        }

    def check_disk_space(self) -> Dict:
        """فحص مساحة القرص وتنظيفها إذا لزم الأمر"""
        disk_usage = psutil.disk_usage('/')
        
        if disk_usage.percent > self.config['disk_threshold']:
            self.clean_old_logs()
            self.clean_cache()
            self.clean_temp_files()
        
        return {
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'used_gb': round(disk_usage.used / (1024**3), 2),
            'free_gb': round(disk_usage.free / (1024**3), 2),
            'percent': disk_usage.percent
        }

    def archive_old_data(self) -> int:
        """أرشفة البيانات القديمة"""
        archived = 0
        historical_dir = self.config['paths']['historical']
        if not os.path.exists(historical_dir):
            return archived

        archive_name = f"historical_data_{datetime.now().strftime('%Y%m%d')}.zip"
        try:
            shutil.make_archive(
                os.path.join(historical_dir, 'archive', archive_name.replace('.zip', '')),
                'zip',
                historical_dir
            )
            archived = 1
            self.logger.info(f"تم إنشاء الأرشيف: {archive_name}")
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء الأرشيف: {str(e)}")

        return archived

    def clean_all(self) -> Dict:
        """تنفيذ جميع عمليات التنظيف"""
        start_time = time.time()
        
        results = {
            'logs_cleaned': self.clean_old_logs(),
            'cache_cleaned': self.clean_cache(),
            'temp_cleaned': self.clean_temp_files(),
            'memory_optimization': self.optimize_memory(),
            'disk_status': self.check_disk_space(),
            'archives_created': self.archive_old_data(),
            'duration_seconds': round(time.time() - start_time, 2)
        }
        
        self.logger.info(f"نتائج التنظيف: {json.dumps(results, ensure_ascii=False, indent=2)}")
        return results

    def get_system_status(self) -> Dict:
        """الحصول على حالة النظام الحالية"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_memory_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        }
