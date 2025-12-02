import { useMemo } from 'react';
import moment from 'moment-jalaali';

import { CALENDAR_COLOR_OPTIONS } from 'src/_mock/_calendar';

// ----------------------------------------------------------------------

export function useEvent(events, selectEventId, selectedRange, openForm) {
  const currentEvent = events.find((event) => event.id === selectEventId);

  const defaultValues = useMemo(
    () => {
      // Helper to convert date to ISO string
      const getDateString = (date) => {
        if (!date) return moment().toISOString();
        if (moment.isMoment(date)) return date.toISOString();
        if (date instanceof Date) return moment(date).toISOString();
        if (typeof date === 'string') {
          const m = moment(date);
          return m.isValid() ? m.toISOString() : moment().toISOString();
        }
        return moment().toISOString();
      };

      return {
        id: '',
        title: '',
        description: '',
        color: CALENDAR_COLOR_OPTIONS[1],
        allDay: false,
        start: selectedRange ? getDateString(selectedRange.start) : moment().toISOString(),
        end: selectedRange ? getDateString(selectedRange.end) : moment().toISOString(),
      };
    },
    [selectedRange]
  );

  if (!openForm) {
    return undefined;
  }

  if (currentEvent || selectedRange) {
    return { ...defaultValues, ...currentEvent };
  }

  return defaultValues;
}
