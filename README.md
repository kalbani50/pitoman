<<<<<<< HEAD
# 🤖 PITOMAN - بوت التداول المتقدم

## 📝 نظرة عامة
PITOMAN هو بوت تداول متقدم يستخدم الذكاء الاصطناعي المتعدد والتعلم العميق للتداول في الأسواق المالية. يتميز بقدرات تحليلية متقدمة وأنظمة ذكاء متعددة تعمل بشكل متكامل.

## ✨ المميزات الرئيسية

### 🧠 نظام التعلم المتقدم
- نظام تجميع نماذج متعدد (Model Ensemble) يشمل:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
- تحديث أوزان النماذج تلقائياً حسب الأداء
- تحليل متعدد الإطارات الزمنية
- نظام اكتشاف أنماط السوق المختلفة

### 📊 تحليل السوق المتقدم
- تحليل الأنظمة السوقية (Market Regimes):
  - السوق العادي
  - السوق المتقلب
  - السوق المتجه
  - السوق المتذبذب
- تحليل العوامل المؤثرة في نجاح التداول
- حساب الثقة في التوقعات
- تحليل الارتباطات بين العوامل المختلفة

### ⚡ تحسينات الأداء
- نظام تخزين مؤقت ذكي للتحليلات
- معالجة متوازية للعمليات
- إدارة معدل الطلبات API
- تحسين استخدام الموارد

### 🛡️ إدارة المخاطر المتقدمة
- تقييم المخاطر الديناميكي
- تحجيم المراكز تلقائياً
- إدارة الرافعة المالية الذكية
- موازنة المحفظة

### 📱 المراقبة والتنبيهات
- مراقبة في الوقت الفعلي
- تنبيهات ذكية
- تقارير أداء تفصيلية
- تتبع الأداء التاريخي

## 🔧 المتطلبات التقنية

### متطلبات النظام
- Windows 10/11 (64-bit)
- Python 3.8+
- CUDA 11.x (للمعالجة على GPU)
- 16GB RAM (الحد الأدنى)
- معالج متعدد النواة
- مساحة 10GB (الحد الأدنى)

### قواعد البيانات
- PostgreSQL 13+
- Redis 6+
- MongoDB 5+ (اختياري)

### المكتبات الرئيسية
- TensorFlow 2.x
- PyTorch 1.x
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- SQLAlchemy

## 📥 التثبيت

### 1. تثبيت المتطلبات الأساسية

#### Windows
```batch
# تثبيت Python
# قم بتحميل وتثبيت Python 3.8+ من python.org

# تثبيت CUDA (للمعالجة على GPU)
# قم بتحميل وتثبيت CUDA Toolkit من موقع NVIDIA

# تثبيت PostgreSQL
# قم بتحميل وتثبيت PostgreSQL من postgresql.org

# تثبيت Redis
# قم بتحميل وتثبيت Redis for Windows
```

#### Linux (اختياري)
```bash
# تثبيت Python
sudo apt update
sudo apt install python3.8 python3-pip

# تثبيت CUDA
# اتبع تعليمات NVIDIA لنظامك

# تثبيت PostgreSQL
sudo apt install postgresql postgresql-contrib

# تثبيت Redis
sudo apt install redis-server
```

### 2. إعداد البيئة
```batch
# استنساخ المشروع
git clone https://github.com/kalbani50/pitoman.git
cd pitoman

# إنشاء بيئة افتراضية
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux

# تثبيت المتطلبات
pip install -r requirements.txt
```

### 3. إعداد قواعد البيانات
```batch
# إعداد PostgreSQL
createdb pitoman_db

# تهيئة قاعدة البيانات
python scripts/setup_database.py

# بدء Redis
net start Redis  # Windows
sudo service redis start  # Linux
```

## 🔧 التكوين

### ملفات التكوين الرئيسية
- `config.yaml`: إعدادات النظام الرئيسية
  - إعدادات نظام التعلم المتقدم
  - تكوين نماذج التحليل
  - إعدادات التخزين المؤقت
  - معلمات إدارة المخاطر
- `.env`: المتغيرات البيئية وبيانات الاعتماد
- `trading_config.yaml`: إعدادات التداول
- `ai_config.yaml`: إعدادات الذكاء الاصطناعي

### تكوين النماذج
```yaml
model_ensemble:
  models:
    - name: random_forest
      n_estimators: 100
    - name: gradient_boosting
      n_estimators: 100
    - name: xgboost
      n_estimators: 100
    - name: lightgbm
      n_estimators: 100
  
  timeframes:
    - short_term
    - medium_term
    - long_term
```

### تكوين التحليل
```yaml
market_analysis:
  cache_ttl: 300  # بالثواني
  update_interval: 60  # بالثواني
  min_confidence: 0.7
  regime_detection: true
```

## 📊 المراقبة والتحليل

### لوحات التحكم
- لوحة التحكم الرئيسية: `http://localhost:8000`
  - عرض حالة التداول الحالية
  - مؤشرات الأداء الرئيسية
  - تحليلات السوق المباشرة
  - إدارة المراكز

- مراقب الأداء: `http://localhost:8001`
  - أداء النماذج
  - إحصائيات التداول
  - تحليل المخاطر
  - سجل العمليات

- تحليلات متقدمة: `http://localhost:8002`
  - تحليل الأنظمة السوقية
  - أداء نماذج التعلم
  - تحليل العوامل المؤثرة
  - تقارير مفصلة

## 🔒 الأمان والموثوقية

### إجراءات الأمان
- تشفير البيانات الحساسة
- مصادقة متعددة العوامل
- مراقبة النشاط المشبوه
- نسخ احتياطي تلقائي

### آليات الحماية
- وقف الطوارئ التلقائي
- حدود المخاطر الديناميكية
- مراقبة صحة النظام
- استعادة تلقائية من الأخطاء

## 🤝 المساهمة
نرحب بمساهماتكم! يرجى اتباع هذه الخطوات:
1. Fork المشروع
2. إنشاء فرع للميزة الجديدة
3. تقديم طلب Pull Request

## 📄 الترخيص
هذا المشروع مرخص تحت MIT License

## 📞 الدعم والتواصل
- GitHub Issues: [https://github.com/kalbani50/pitoman](https://github.com/kalbani50/pitoman)
- البريد الإلكتروني: PTIOMAN@LIVE.COM

=======
# pitoman
Trading bot for Binance and OKX runs as smoothly as possible without human intervention after the first run - not yet tested
>>>>>>> 8ed59750948363050da2feadbc0d41515a3dfbe0
