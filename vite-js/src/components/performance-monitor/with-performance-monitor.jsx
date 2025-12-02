import { Profiler } from 'react';

import { PerformanceMonitor } from './performance-monitor';

/**
 * Higher-Order Component برای wrap کردن کامپوننت‌ها و ردیابی عملکرد
 * 
 * @param {React.Component} Component - کامپوننت مورد نظر
 * @param {Object} options - تنظیمات
 * @param {string} options.componentName - نام کامپوننت
 * @param {boolean} options.showMonitor - نمایش مانیتور
 * @param {string} options.position - موقعیت مانیتور
 * @returns {React.Component} - کامپوننت wrapped شده
 */
export function withPerformanceMonitor(Component, options = {}) {
  const {
    componentName = Component.displayName || Component.name || 'Unknown',
    showMonitor = true,
    position = 'bottom-right',
  } = options;

  const WrappedComponent = (props) => {
    const onRenderCallback = (id, phase, actualDuration, baseDuration, startTime, commitTime) => {
      if (process.env.NODE_ENV === 'development') {
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
      <>
        <Profiler id={componentName} onRender={onRenderCallback}>
          <Component {...props} />
        </Profiler>
        {showMonitor && <PerformanceMonitor componentName={componentName} position={position} />}
      </>
    );
  };

  WrappedComponent.displayName = `withPerformanceMonitor(${componentName})`;

  return WrappedComponent;
}


