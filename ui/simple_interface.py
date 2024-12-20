"""
واجهة مستخدم مبسطة لـ Binance و OKX
"""

import tkinter as tk
from tkinter import ttk
import json
from typing import Dict, List
import logging

class SimpleTraderUI:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.root = tk.Tk()
        self.root.title("Binance & OKX Trading Bot")
        self.setup_ui()
        
    def setup_ui(self):
        """إعداد الواجهة الرئيسية"""
        # إطار التداول
        self.trading_frame = ttk.LabelFrame(self.root, text="التداول")
        self.trading_frame.pack(padx=10, pady=5, fill="x")
        
        # اختيار المنصة
        ttk.Label(self.trading_frame, text="المنصة:").pack()
        self.exchange_var = tk.StringVar(value="binance")
        ttk.Radiobutton(self.trading_frame, text="Binance", variable=self.exchange_var, value="binance").pack()
        ttk.Radiobutton(self.trading_frame, text="OKX", variable=self.exchange_var, value="okx").pack()
        
        # إعدادات التداول الأساسية
        self.setup_trading_settings()
        
        # عرض الحالة
        self.setup_status_display()
        
    def setup_trading_settings(self):
        """إعداد إعدادات التداول"""
        settings_frame = ttk.LabelFrame(self.root, text="الإعدادات")
        settings_frame.pack(padx=10, pady=5, fill="x")
        
        # الزوج
        ttk.Label(settings_frame, text="الزوج:").pack()
        self.pair_var = tk.StringVar(value="BTC/USDT")
        ttk.Entry(settings_frame, textvariable=self.pair_var).pack()
        
        # الكمية
        ttk.Label(settings_frame, text="الكمية:").pack()
        self.amount_var = tk.StringVar(value="0.001")
        ttk.Entry(settings_frame, textvariable=self.amount_var).pack()
        
        # الرافعة المالية
        ttk.Label(settings_frame, text="الرافعة:").pack()
        self.leverage_var = tk.StringVar(value="10")
        ttk.Entry(settings_frame, textvariable=self.leverage_var).pack()
        
    def setup_status_display(self):
        """إعداد عرض الحالة"""
        status_frame = ttk.LabelFrame(self.root, text="الحالة")
        status_frame.pack(padx=10, pady=5, fill="x")
        
        # حالة الاتصال
        self.connection_label = ttk.Label(status_frame, text="متصل")
        self.connection_label.pack()
        
        # الرصيد
        self.balance_label = ttk.Label(status_frame, text="الرصيد: 0 USDT")
        self.balance_label.pack()
        
        # الربح/الخسارة
        self.pnl_label = ttk.Label(status_frame, text="الربح/الخسارة: 0 USDT")
        self.pnl_label.pack()

class QuickTradePanel:
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="التداول السريع")
        self.frame.pack(padx=10, pady=5, fill="x")
        self.setup_panel()
        
    def setup_panel(self):
        """إعداد لوحة التداول السريع"""
        # زر الشراء
        self.buy_button = ttk.Button(self.frame, text="شراء", command=self.quick_buy)
        self.buy_button.pack(side="left", padx=5)
        
        # زر البيع
        self.sell_button = ttk.Button(self.frame, text="بيع", command=self.quick_sell)
        self.sell_button.pack(side="right", padx=5)
        
    def quick_buy(self):
        """تنفيذ شراء سريع"""
        # تنفيذ أمر شراء سريع
        pass
        
    def quick_sell(self):
        """تنفيذ بيع سريع"""
        # تنفيذ أمر بيع سريع
        pass

class RiskControlPanel:
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="إدارة المخاطر")
        self.frame.pack(padx=10, pady=5, fill="x")
        self.setup_panel()
        
    def setup_panel(self):
        """إعداد لوحة إدارة المخاطر"""
        # وقف الخسارة
        ttk.Label(self.frame, text="وقف الخسارة (%):").pack()
        self.stop_loss_var = tk.StringVar(value="2")
        ttk.Entry(self.frame, textvariable=self.stop_loss_var).pack()
        
        # جني الأرباح
        ttk.Label(self.frame, text="جني الأرباح (%):").pack()
        self.take_profit_var = tk.StringVar(value="3")
        ttk.Entry(self.frame, textvariable=self.take_profit_var).pack()
        
        # الحد الأقصى للمخاطرة
        ttk.Label(self.frame, text="الحد الأقصى للمخاطرة (%):").pack()
        self.max_risk_var = tk.StringVar(value="1")
        ttk.Entry(self.frame, textvariable=self.max_risk_var).pack()
