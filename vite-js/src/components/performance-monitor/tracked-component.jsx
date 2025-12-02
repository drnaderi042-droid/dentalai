import { Profiler , useState, useEffect } from 'react';

import { usePerformanceMonitor } from 'src/hooks/use-performance-monitor';

import { usePerformanceContext } from 'src/contexts/performance-context';

// ----------------------------------------------------------------------

/**
 * کامپوننت wrapper برای ردیابی عملکرد هر کامپوننت
 * 
 * @param {React.Component} children - کامپوننت فرزند
 * @param {string} componentName - نام کامپوننت
 * @param {Object} options - تنظیمات
 */
export function TrackedComponent({ children, componentName, options = {} }) {
  const { registerComponent, registerProfilerData, getAllComponents } = usePerformanceContext();
  const [profilerDuration, setProfilerDuration] = useState(0);
  
  const metrics = usePerformanceMonitor(componentName, {
    interval: options.interval || 1000,
    trackMemory: options.trackMemory !== false,
    trackCPU: options.trackCPU !== false,
    profilerDuration,
  });

  // ثبت متریک‌ها در context
  useEffect(() => {
    registerComponent(componentName, metrics);
  }, [componentName, metrics, registerComponent]);

  // به‌روزرسانی محاسبات وقتی profilerDuration تغییر می‌کند
  useEffect(() => {
    if (profilerDuration > 0) {
      // Force update metrics when profiler duration changes
      // This will trigger calculateMetrics in usePerformanceMonitor
    }
  }, [profilerDuration]);

  // Callback برای React Profiler
  const onRenderCallback = (id, phase, actualDuration, baseDuration, startTime, commitTime) => {
    const profilerData = {
      phase,
      actualDuration,
      baseDuration,
      startTime,
      commitTime,
      interactions: [],
    };

    // به‌روزرسانی زمان render برای محاسبه متریک‌ها
    // فقط اگر actualDuration معتبر باشد (بزرگتر از 0)
    if (actualDuration > 0) {
      setProfilerDuration(actualDuration);
    }

    registerProfilerData(componentName, profilerData);

    if (process.env.NODE_ENV === 'development' && options.logProfiler) {
      console.log(`[Profiler - ${componentName}]`, {
        phase,
        actualDuration: `${actualDuration.toFixed(2)}ms`,
        baseDuration: `${baseDuration.toFixed(2)}ms`,
        startTime: `${startTime.toFixed(2)}ms`,
        commitTime: `${commitTime.toFixed(2)}ms`,
      });
    }
  };

  return (
    <Profiler id={componentName} onRender={onRenderCallback}>
      {children}
    </Profiler>
  );
}

