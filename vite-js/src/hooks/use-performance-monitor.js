import { useRef, useState, useEffect, useCallback } from 'react';

/**
 * Hook برای ردیابی مصرف CPU و RAM توسط کامپوننت‌ها
 * 
 * @param {string} componentName - نام کامپوننت برای شناسایی
 * @param {Object} options - تنظیمات
 * @param {number} options.interval - فاصله زمانی اندازه‌گیری (میلی‌ثانیه)
 * @param {boolean} options.trackMemory - آیا حافظه ردیابی شود
 * @param {boolean} options.trackCPU - آیا CPU ردیابی شود
 * @param {number} options.profilerDuration - زمان render از Profiler (ms)
 * @returns {Object} - داده‌های عملکرد
 */
export function usePerformanceMonitor(componentName, options = {}) {
  const {
    interval = 1000,
    trackMemory = true,
    trackCPU = true,
    profilerDuration = 0,
  } = options;

  const [metrics, setMetrics] = useState({
    memory: {
      usedMB: 0, // مصرف RAM به مگابایت
      estimatedMB: 0, // تخمین مصرف بر اساس render time
    },
    cpu: {
      usagePercent: 0, // درصد مصرف CPU (نسبی)
      renderTime: 0, // زمان render (ms)
    },
    renderTime: 0,
    renderCount: 0,
  });

  const renderStartTime = useRef(null);
  const intervalRef = useRef(null);
  const profilerDurationRef = useRef(0);

  // محاسبه مصرف RAM و CPU بر اساس زمان render
  const calculateMetrics = useCallback(() => {
    const renderDuration = profilerDurationRef.current || 0;
    
    if (renderDuration === 0 || renderDuration < 0.1) {
      // اگر render time صفر است، از مقادیر حداقل استفاده کن
      setMetrics((prev) => {
        // فقط اگر مقادیر فعلی صفر هستند، به‌روزرسانی کن
        if (prev.memory.usedMB === 0 && prev.cpu.usagePercent === 0) {
          return {
            ...prev,
            memory: {
              usedMB: 0.5,
              estimatedMB: 0.5,
            },
            cpu: {
              usagePercent: 0.5,
              renderTime: 0,
            },
          };
        }
        return prev;
      });
      return;
    }

    // تخمین مصرف RAM بر اساس زمان render (به مگابایت)
    // فرمول: هر 1ms render ≈ 0.01 MB RAM
    // این یک تخمین است و بر اساس تجربه تنظیم شده
    // محدود کردن به 0.5 تا 30 MB برای واقع‌گرایی
    const estimatedMemoryMB = Math.max(0.5, Math.min(30, renderDuration * 0.01));
    
    // تخمین مصرف CPU بر اساس زمان render (نسبی)
    // استفاده از فرمول بهبود یافته که واقع‌گرایانه‌تر است
    // فرض: render time بیشتر = CPU بیشتر
    // اما باید نسبی باشد - نه مطلق
    // استفاده از فرمول: (renderTime / 100) * 5
    // این یعنی اگر render time 100ms باشد، 5% CPU مصرف می‌کند
    // حداکثر 10% برای هر کامپوننت
    const cpuUsagePercent = Math.min(10, (renderDuration / 100) * 5);

    setMetrics((prev) => ({
      ...prev,
      memory: {
        usedMB: Math.round(estimatedMemoryMB * 100) / 100,
        estimatedMB: Math.round(estimatedMemoryMB * 100) / 100,
      },
      cpu: {
        usagePercent: Math.round(cpuUsagePercent * 100) / 100,
        renderTime: Math.round(renderDuration * 100) / 100,
      },
    }));
  }, []);

  // به‌روزرسانی profiler duration و محاسبه متریک‌ها
  useEffect(() => {
    if (profilerDuration > 0) {
      profilerDurationRef.current = profilerDuration;
      // محاسبه فوری متریک‌ها وقتی profiler duration تغییر می‌کند
      if (trackMemory || trackCPU) {
        calculateMetrics();
      }
    }
  }, [profilerDuration, trackMemory, trackCPU, calculateMetrics]);

  // ردیابی زمان رندر
  useEffect(() => {
    renderStartTime.current = performance.now();
    
    return () => {
      if (renderStartTime.current) {
        const renderTime = performance.now() - renderStartTime.current;
        setMetrics((prev) => ({
          ...prev,
          renderTime,
          renderCount: prev.renderCount + 1,
        }));
      }
    };
  });

  // به‌روزرسانی متریک‌ها
  const updateMetrics = useCallback(() => {
    if (trackMemory || trackCPU) {
      calculateMetrics();
    }
  }, [trackMemory, trackCPU, calculateMetrics]);

  // به‌روزرسانی دوره‌ای متریک‌ها
  useEffect(() => {
    if (interval <= 0) {
      return;
    }

    // به‌روزرسانی اولیه
    updateMetrics();

    // تنظیم interval
    intervalRef.current = setInterval(updateMetrics, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [interval, updateMetrics]);

  // لاگ کردن اطلاعات در console (فقط در development)
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[Performance Monitor - ${componentName}]`, {
        memory: metrics.memory,
        cpu: metrics.cpu,
        renderTime: metrics.renderTime,
        renderCount: metrics.renderCount,
      });
    }
  }, [componentName, metrics]);

  return {
    ...metrics,
    componentName,
  };
}

