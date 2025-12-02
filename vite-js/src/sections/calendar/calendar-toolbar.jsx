import Stack from '@mui/material/Stack';
import Badge from '@mui/material/Badge';
import Button from '@mui/material/Button';
import MenuList from '@mui/material/MenuList';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import LinearProgress from '@mui/material/LinearProgress';

import { fDateJalali } from 'src/utils/format-time';

import { Iconify } from 'src/components/iconify';
import { usePopover, CustomPopover } from 'src/components/custom-popover';

// ----------------------------------------------------------------------

const VIEW_OPTIONS = [
  { value: 'dayGridMonth', label: 'ماه', icon: 'mingcute:calendar-month-line' },
  { value: 'timeGridWeek', label: 'هفته', icon: 'mingcute:calendar-week-line' },
  { value: 'timeGridDay', label: 'روز', icon: 'mingcute:calendar-day-line' },
  { value: 'listWeek', label: 'برنامه', icon: 'fluent:calendar-agenda-24-regular' },
];

// ----------------------------------------------------------------------

export function CalendarToolbar({
  date,
  view,
  loading,
  onToday,
  canReset,
  onNextDate,
  onPrevDate,
  onChangeView,
  onOpenFilters,
}) {
  const popover = usePopover();

  const selectedItem = VIEW_OPTIONS.filter((item) => item.value === view)[0];

  return (
    <>
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{ 
          p: { xs: 1.5, sm: 2.5 }, 
          pr: { xs: 1, sm: 2 }, 
          position: 'relative',
          gap: { xs: 0.5, sm: 1 }
        }}
      >
        {/* Desktop: Button with label */}
        <Button
          size="small"
          color="inherit"
          onClick={popover.onOpen}
          startIcon={<Iconify icon={selectedItem.icon} />}
          endIcon={<Iconify icon="eva:arrow-ios-downward-fill" sx={{ ml: -0.5 }} />}
          sx={{ display: { xs: 'none', sm: 'inline-flex' } }}
        >
          {selectedItem.label}
        </Button>

        {/* Mobile: IconButton */}
        <IconButton
          onClick={popover.onOpen}
          sx={{ display: { xs: 'inline-flex', sm: 'none' } }}
        >
          <Iconify icon={selectedItem.icon} />
        </IconButton>

        <Stack direction="row" alignItems="center" spacing={{ xs: 0.5, sm: 1 }}>
          <IconButton onClick={onPrevDate} size="small">
            <Iconify icon="eva:arrow-ios-forward-fill" />
          </IconButton>

          <Typography 
            variant="h6" 
            sx={{ 
              fontSize: { xs: '0.875rem', sm: '1.25rem' },
              minWidth: { xs: '120px', sm: 'auto' },
              textAlign: 'center'
            }}
          >
            {fDateJalali(date, 'jYYYY/jMM/jDD')}
          </Typography>

          <IconButton onClick={onNextDate} size="small">
            <Iconify icon="eva:arrow-ios-back-fill" />
          </IconButton>
        </Stack>

        <Stack direction="row" alignItems="center" spacing={{ xs: 0.5, sm: 1 }}>
          <Button 
            size="small" 
            color="error" 
            variant="contained" 
            onClick={onToday}
            sx={{ 
              fontSize: { xs: '0.75rem', sm: '0.875rem' },
              px: { xs: 1, sm: 2 }
            }}
          >
            امروز
          </Button>

          <IconButton onClick={onOpenFilters} size="small">
            <Badge color="error" variant="dot" invisible={!canReset}>
              <Iconify icon="ic:round-filter-list" />
            </Badge>
          </IconButton>
        </Stack>

        {loading && (
          <LinearProgress
            color="inherit"
            sx={{
              left: 0,
              width: 1,
              height: 2,
              bottom: 0,
              borderRadius: 0,
              position: 'absolute',
            }}
          />
        )}
      </Stack>

      <CustomPopover
        open={popover.open}
        anchorEl={popover.anchorEl}
        onClose={popover.onClose}
        slotProps={{ arrow: { placement: 'top-left' } }}
      >
        <MenuList>
          {VIEW_OPTIONS.map((viewOption) => (
            <MenuItem
              key={viewOption.value}
              selected={viewOption.value === view}
              onClick={() => {
                popover.onClose();
                onChangeView(viewOption.value);
              }}
            >
              <Iconify icon={viewOption.icon} />
              {viewOption.label}
            </MenuItem>
          ))}
        </MenuList>
      </CustomPopover>
    </>
  );
}
