import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

// ----------------------------------------------------------------------

export function SearchNotFound({ query, sx, ...other }) {
  if (!query) {
    return (
      <Typography variant="body2" sx={sx}>
        متنی وارد کنید
      </Typography>
    );
  }

  return (
    <Box sx={{ textAlign: 'center', borderRadius: 1.5, ...sx }} {...other}>
      <Box sx={{ mb: 1, typography: 'h6' }}>یافت نشد</Box>

      <Typography variant="body2">
        هیچ نتیجه ای برای &nbsp;
        <strong>{`"${query}"`}</strong> یافت نشد
        .
        <br /> لطفا عبارت کوتاه تری وارد کنید
      </Typography>
    </Box>
  );
}
