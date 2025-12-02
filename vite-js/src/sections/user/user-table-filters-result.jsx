import { useCallback } from 'react';

import Chip from '@mui/material/Chip';

import { chipProps, FiltersBlock, FiltersResult } from 'src/components/filters-result';

// ----------------------------------------------------------------------

export function UserTableFiltersResult({ filters, onResetPage, totalResults, sx }) {
  const handleRemoveKeyword = useCallback(() => {
    onResetPage();
    filters.setState({ name: '' });
  }, [filters, onResetPage]);

  const handleRemoveStatus = useCallback(() => {
    onResetPage();
    filters.setState({ status: 'all' });
  }, [filters, onResetPage]);

  const handleRemoveRole = useCallback(
    (inputValue) => {
      const newValue = filters.state.role.filter((item) => item !== inputValue);

      onResetPage();
      filters.setState({ role: newValue });
    },
    [filters, onResetPage]
  );

  const handleReset = useCallback(() => {
    onResetPage();
    filters.onResetState();
  }, [filters, onResetPage]);

  return (
    <FiltersResult totalResults={totalResults} onReset={handleReset} sx={sx}>
      <FiltersBlock label="وضعیت:" isShow={filters.state.status !== 'all'}>
        <Chip
          {...chipProps}
          label={
            filters.state.status === 'active' ? 'فعال' :
            filters.state.status === 'pending' ? 'در انتظار' :
            filters.state.status === 'banned' ? 'مسدود' :
            filters.state.status
          }
          onDelete={handleRemoveStatus}
        />
      </FiltersBlock>

      <FiltersBlock label="نقش:" isShow={!!filters.state.role.length}>
        {filters.state.role.map((item) => {
          const roleMap = {
            'DOCTOR': 'دکتر',
            'PATIENT': 'بیمار',
            'ADMIN': 'ادمین',
          };
          return (
            <Chip 
              {...chipProps} 
              key={item} 
              label={roleMap[item] || item} 
              onDelete={() => handleRemoveRole(item)} 
            />
          );
        })}
      </FiltersBlock>

      <FiltersBlock label="کلمه کلیدی:" isShow={!!filters.state.name}>
        <Chip {...chipProps} label={filters.state.name} onDelete={handleRemoveKeyword} />
      </FiltersBlock>
    </FiltersResult>
  );
}
