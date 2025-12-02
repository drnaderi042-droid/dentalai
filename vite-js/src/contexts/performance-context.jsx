import { useRef, useMemo, useState, useContext, useCallback, createContext } from 'react';

// ----------------------------------------------------------------------

const PerformanceContext = createContext(null);

// ----------------------------------------------------------------------

/**
 * Provider برای مدیریت ردیابی عملکرد تمام کامپوننت‌ها
 */
export function PerformanceProvider({ children }) {
  const [components, setComponents] = useState(new Map());
  const profilerDataRef = useRef(new Map());

  // ثبت کامپوننت جدید
  const registerComponent = useCallback((componentName, metrics) => {
    setComponents((prev) => {
      const newMap = new Map(prev);
      newMap.set(componentName, {
        ...metrics,
        componentName,
        lastUpdate: Date.now(),
      });
      return newMap;
    });
  }, []);

  // ثبت داده‌های Profiler
  const registerProfilerData = useCallback((componentName, profilerData) => {
    profilerDataRef.current.set(componentName, {
      ...profilerData,
      timestamp: Date.now(),
    });
  }, []);

  // دریافت داده‌های یک کامپوننت
  const getComponentData = useCallback((componentName) => components.get(componentName) || null, [components]);

  // دریافت تمام کامپوننت‌ها
  const getAllComponents = useCallback(() => Array.from(components.values()), [components]);

  // دریافت داده‌های Profiler
  const getProfilerData = useCallback((componentName) => profilerDataRef.current.get(componentName) || null, []);

  // پاک کردن داده‌های یک کامپوننت
  const unregisterComponent = useCallback((componentName) => {
    setComponents((prev) => {
      const newMap = new Map(prev);
      newMap.delete(componentName);
      return newMap;
    });
    profilerDataRef.current.delete(componentName);
  }, []);

  // پاک کردن تمام داده‌ها
  const clearAll = useCallback(() => {
    setComponents(new Map());
    profilerDataRef.current.clear();
  }, []);

  const value = useMemo(() => ({
    components,
    registerComponent,
    registerProfilerData,
    getComponentData,
    getAllComponents,
    getProfilerData,
    unregisterComponent,
    clearAll,
  }), [components, registerComponent, registerProfilerData, getComponentData, getAllComponents, getProfilerData, unregisterComponent, clearAll]);

  return <PerformanceContext.Provider value={value}>{children}</PerformanceContext.Provider>;
}

// ----------------------------------------------------------------------

/**
 * Hook برای دسترسی به Performance Context
 */
export function usePerformanceContext() {
  const context = useContext(PerformanceContext);
  if (!context) {
    throw new Error('usePerformanceContext must be used within PerformanceProvider');
  }
  return context;
}
