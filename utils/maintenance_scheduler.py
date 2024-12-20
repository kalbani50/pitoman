import schedule
import time
import logging
from datetime import datetime
from typing import Dict, Callable
import json
from .resource_cleaner import ResourceCleaner

class MaintenanceScheduler:
    """جدولة وإدارة عمليات الصيانة الدورية"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.cleaner = ResourceCleaner(config)
        self.config = config or {
            'schedules': {
                'memory_check': '*/30 * * * *',      # كل 30 دقيقة
                'disk_check': '0 */2 * * *',         # كل ساعتين
                'log_cleanup': '0 0 * * *',          # يومياً
                'cache_cleanup': '0 */4 * * *',      # كل 4 ساعات
                'temp_cleanup': '0 */6 * * *',       # كل 6 ساعات
                'full_cleanup': '0 0 * * 0',         # أسبوعياً
                'archiving': '0 0 1 * *'             # شهرياً
            },
            'alerts': {
                'memory_threshold': 90,
                'disk_threshold': 85,
                'notification_webhook': None
            }
        }
        
    def setup_schedules(self):
        """إعداد جداول الصيانة الدورية"""
        # فحص الذاكرة
        schedule.every(30).minutes.do(self._run_task, 
            task=self.cleaner.optimize_memory,
            task_name="فحص الذاكرة")

        # فحص القرص
        schedule.every(2).hours.do(self._run_task,
            task=self.cleaner.check_disk_space,
            task_name="فحص القرص")

        # تنظيف السجلات
        schedule.every().day.at("00:00").do(self._run_task,
            task=self.cleaner.clean_old_logs,
            task_name="تنظيف السجلات")

        # تنظيف الكاش
        schedule.every(4).hours.do(self._run_task,
            task=self.cleaner.clean_cache,
            task_name="تنظيف الكاش")

        # تنظيف الملفات المؤقتة
        schedule.every(6).hours.do(self._run_task,
            task=self.cleaner.clean_temp_files,
            task_name="تنظيف الملفات المؤقتة")

        # تنظيف شامل
        schedule.every().sunday.at("00:00").do(self._run_task,
            task=self.cleaner.clean_all,
            task_name="تنظيف شامل")

        # أرشفة البيانات
        schedule.every().month.at("00:00").do(self._run_task,
            task=self.cleaner.archive_old_data,
            task_name="أرشفة البيانات")

    def _run_task(self, task: Callable, task_name: str) -> Dict:
        """تنفيذ المهمة مع التسجيل والتنبيهات"""
        try:
            self.logger.info(f"بدء {task_name}")
            start_time = time.time()
            
            result = task()
            
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"اكتمال {task_name} في {duration} ثانية")
            
            # فحص النتائج وإرسال تنبيهات إذا لزم الأمر
            self._check_alerts(result, task_name)
            
            return {
                'task': task_name,
                'status': 'success',
                'duration': duration,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"خطأ في {task_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'task': task_name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _check_alerts(self, result: Dict, task_name: str):
        """فحص النتائج وإرسال تنبيهات إذا تجاوزت العتبات"""
        if isinstance(result, dict):
            # فحص استخدام الذاكرة
            if 'memory_percent' in result and \
               result['memory_percent'] > self.config['alerts']['memory_threshold']:
                self._send_alert(f"تحذير: استخدام الذاكرة مرتفع ({result['memory_percent']}%)")

            # فحص مساحة القرص
            if 'disk_percent' in result and \
               result['disk_percent'] > self.config['alerts']['disk_threshold']:
                self._send_alert(f"تحذير: مساحة القرص منخفضة (متبقي {100 - result['disk_percent']}%)")

    def _send_alert(self, message: str):
        """إرسال تنبيه عبر Webhook أو تسجيله"""
        self.logger.warning(message)
        if self.config['alerts']['notification_webhook']:
            # هنا يمكن إضافة كود لإرسال التنبيهات عبر Webhook
            pass

    def start(self):
        """بدء جدولة الصيانة"""
        self.setup_schedules()
        self.logger.info("بدء جدولة الصيانة")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # انتظار دقيقة واحدة
            except Exception as e:
                self.logger.error(f"خطأ في دورة الجدولة: {str(e)}")
                time.sleep(300)  # انتظار 5 دقائق في حالة الخطأ

    def get_next_runs(self) -> Dict:
        """الحصول على توقيت المهام القادمة"""
        next_runs = {}
        for job in schedule.jobs:
            next_runs[job.job_func.__name__] = {
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'last_run': job.last_run.isoformat() if job.last_run else None
            }
        return next_runs
