import { useEffect } from 'react';
import moment from 'moment-jalaali';
import Calendar from '@fullcalendar/react'; // => request placed at the top
import listPlugin from '@fullcalendar/list';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import timelinePlugin from '@fullcalendar/timeline';
import faLocale from '@fullcalendar/core/locales/fa';
import interactionPlugin from '@fullcalendar/interaction';

import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import { useTheme } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';

import { useBoolean } from 'src/hooks/use-boolean';
import { useSetState } from 'src/hooks/use-set-state';

import { fIsAfter, fIsBetween } from 'src/utils/format-time';
import { toPersianDigits } from 'src/utils/convert-to-persian-digits';

import { DashboardContent } from 'src/layouts/dashboard';
import { CALENDAR_COLOR_OPTIONS } from 'src/_mock/_calendar';
import { updateEvent, useGetEvents } from 'src/actions/calendar';

import { Iconify } from 'src/components/iconify';

import { StyledCalendar } from '../styles';
import { useEvent } from '../hooks/use-event';
import { CalendarForm } from '../calendar-form';
import { useCalendar } from '../hooks/use-calendar';
import { CalendarFilters } from '../calendar-filters';
import { CalendarToolbar } from '../calendar-toolbar';
import { CalendarFiltersResult } from '../calendar-filters-result';

// Ensure Persian locale is loaded for moment-jalaali
moment.loadPersian({ dialect: 'persian-modern' });

// ----------------------------------------------------------------------

export function CalendarView() {
  const theme = useTheme();

  const openFilters = useBoolean();

  const { events, eventsLoading } = useGetEvents();

  const filters = useSetState({
    colors: [],
    startDate: null,
    endDate: null,
  });

  const dateError = fIsAfter(filters.state.startDate, filters.state.endDate);

  const {
    calendarRef,
    //
    view,
    date,
    //
    onDatePrev,
    onDateNext,
    onDateToday,
    onDropEvent,
    onChangeView,
    onSelectRange,
    onClickEvent,
    onResizeEvent,
    onInitialView,
    //
    openForm,
    onOpenForm,
    onCloseForm,
    //
    selectEventId,
    selectedRange,
    //
    onClickEventInFilters,
  } = useCalendar();

  const currentEvent = useEvent(events, selectEventId, selectedRange, openForm);

  useEffect(() => {
    onInitialView();
  }, [onInitialView]);


  const canReset =
    filters.state.colors.length > 0 || (!!filters.state.startDate && !!filters.state.endDate);

  const dataFiltered = applyFilter({ inputData: events, filters: filters.state, dateError });

  const renderResults = (
    <CalendarFiltersResult
      filters={filters}
      totalResults={dataFiltered.length}
      sx={{ mb: { xs: 3, md: 5 } }}
    />
  );

  const flexProps = { flex: '1 1 auto', display: 'flex', flexDirection: 'column' };

  return (
    <>
      <DashboardContent maxWidth="xl" sx={{ ...flexProps }}>
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="space-between"
          sx={{ mb: { xs: 3, md: 5 } }}
        >
          <Typography variant="h4">تقویم</Typography>
          <Button
            variant="contained"
            startIcon={<Iconify icon="mingcute:add-line" />}
            onClick={onOpenForm}
          >
            رویداد جدید
          </Button>
        </Stack>

        {canReset && renderResults}

        <Card sx={{ ...flexProps, minHeight: '50vh' }}>
          <StyledCalendar sx={{ ...flexProps, '.fc.fc-media-screen': { flex: '1 1 auto' } }}>
            <CalendarToolbar
              date={date}
              view={view}
              canReset={canReset}
              loading={eventsLoading}
              onNextDate={onDateNext}
              onPrevDate={onDatePrev}
              onToday={onDateToday}
              onChangeView={onChangeView}
              onOpenFilters={openFilters.onTrue}
            />

            <Calendar
              weekends
              editable
              droppable
              selectable
              rerenderDelay={10}
              allDayMaintainDuration
              eventResizableFromStart
              ref={calendarRef}
              initialDate={date}
              initialView={view}
              dayMaxEventRows={3}
              eventDisplay="block"
              events={dataFiltered}
              headerToolbar={false}
              locale={faLocale}
              direction="rtl"
              select={onSelectRange}
              eventClick={onClickEvent}
              aspectRatio={3}
              views={{
                timeGridWeek: {
                  dayHeaderContent: (arg) => {
                    // Map weekday index to Persian single character
                    // JavaScript Date.getDay(): 0=Sunday, 1=Monday, ..., 6=Saturday
                    // Persian week starts with Saturday (شنبه)
                    const weekdayCharMap = {
                      6: 'ش', // شنبه (Saturday) - JavaScript getDay() returns 6
                      0: 'ی', // یکشنبه (Sunday) - JavaScript getDay() returns 0
                      1: 'د', // دوشنبه (Monday) - JavaScript getDay() returns 1
                      2: 'س', // سه‌شنبه (Tuesday) - JavaScript getDay() returns 2
                      3: 'چ', // چهارشنبه (Wednesday) - JavaScript getDay() returns 3
                      4: 'پ', // پنج‌شنبه (Thursday) - JavaScript getDay() returns 4
                      5: 'ج', // جمعه (Friday) - JavaScript getDay() returns 5
                    };
                    
                    try {
                      // Get weekday from JavaScript Date (0=Sunday, 6=Saturday)
                      const weekdayIndex = arg.date.getDay();
                      
                      // Get single character for day name
                      const singleChar = weekdayCharMap[weekdayIndex] || '?';
                      
                      // Convert Gregorian date to Jalali (Persian) using moment-jalaali
                      const jalaliMoment = moment(arg.date);
                      
                      // Get Jalali day number
                      const jalaliDay = jalaliMoment.jDate();
                      
                      // Convert to Persian digits
                      const persianDayNumber = toPersianDigits(jalaliDay.toString());
                      
                      // Return string with single char day name and Jalali date
                      return `${singleChar} ${persianDayNumber}`;
                    } catch (error) {
                      // Fallback
                      try {
                        const weekdayIndex = arg.date.getDay();
                        const singleChar = weekdayCharMap[weekdayIndex] || '?';
                        const jalaliMoment = moment(arg.date);
                        const jalaliDay = jalaliMoment.jDate();
                        const persianDayNumber = toPersianDigits(jalaliDay.toString());
                        return `${singleChar} ${persianDayNumber}`;
                      } catch (e) {
                        // Final fallback
                        const dayNum = arg.date.getDate();
                        return toPersianDigits(dayNum.toString());
                      }
                    }
                  },
                },
              }}
              eventDrop={(arg) => {
                onDropEvent(arg, updateEvent);
              }}
              eventResize={(arg) => {
                onResizeEvent(arg, updateEvent);
              }}
              plugins={[
                listPlugin,
                dayGridPlugin,
                timelinePlugin,
                timeGridPlugin,
                interactionPlugin,
              ]}
            />
          </StyledCalendar>
        </Card>
      </DashboardContent>

      <Dialog
        fullWidth
        maxWidth="xs"
        open={openForm}
        onClose={onCloseForm}
        transitionDuration={{
          enter: theme.transitions.duration.shortest,
          exit: theme.transitions.duration.shortest - 80,
        }}
        PaperProps={{
          sx: {
            display: 'flex',
            overflow: 'hidden',
            flexDirection: 'column',
            '& form': { minHeight: 0, display: 'flex', flex: '1 1 auto', flexDirection: 'column' },
          },
        }}
      >
        <DialogTitle sx={{ minHeight: 76 }}>
          {openForm && <> {currentEvent?.id ? 'ویرایش' : 'افزودن'} رویداد</>}
        </DialogTitle>

        <CalendarForm
          currentEvent={currentEvent}
          colorOptions={CALENDAR_COLOR_OPTIONS}
          onClose={onCloseForm}
        />
      </Dialog>

      <CalendarFilters
        events={events}
        filters={filters}
        canReset={canReset}
        dateError={dateError}
        open={openFilters.value}
        onClose={openFilters.onFalse}
        onClickEvent={onClickEventInFilters}
        colorOptions={CALENDAR_COLOR_OPTIONS}
      />
    </>
  );
}

function applyFilter({ inputData, filters, dateError }) {
  const { colors, startDate, endDate } = filters;

  const stabilizedThis = inputData.map((el, index) => [el, index]);

  inputData = stabilizedThis.map((el) => el[0]);

  if (colors.length) {
    inputData = inputData.filter((event) => colors.includes(event.color));
  }

  if (!dateError) {
    if (startDate && endDate) {
      inputData = inputData.filter((event) => fIsBetween(event.start, startDate, endDate));
    }
  }

  return inputData;
}
