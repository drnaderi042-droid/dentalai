import { TrackedComponent } from './tracked-component';

/**
 * HOC برای wrap کردن کامپوننت‌ها و ردیابی عملکرد
 * 
 * @param {React.Component} Component - کامپوننت مورد نظر
 * @param {Object} options - تنظیمات
 * @param {string} options.componentName - نام کامپوننت
 * @param {number} options.interval - فاصله به‌روزرسانی
 * @param {boolean} options.trackMemory - ردیابی حافظه
 * @param {boolean} options.trackCPU - ردیابی CPU
 * @param {boolean} options.logProfiler - لاگ کردن Profiler
 * @returns {React.Component} - کامپوننت wrapped شده
 */
export function withTrackedComponent(Component, options = {}) {
  const {
    componentName = Component.displayName || Component.name || 'Unknown',
    interval = 1000,
    trackMemory = true,
    trackCPU = true,
    logProfiler = false,
  } = options;

  const WrappedComponent = (props) => (
      <TrackedComponent
        componentName={componentName}
        options={{
          interval,
          trackMemory,
          trackCPU,
          logProfiler,
        }}
      >
        <Component {...props} />
      </TrackedComponent>
    );

  WrappedComponent.displayName = `withTrackedComponent(${componentName})`;

  return WrappedComponent;
}


