// @mui
import PropTypes from 'prop-types';
import { Card, Typography } from '@mui/material';

// ----------------------------------------------------------------------

AppWidgetSummaryNoIcon.propTypes = {
  color: PropTypes.string,
  icon: PropTypes.string,
  title: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
  sx: PropTypes.object,
  isText: PropTypes.bool,
};

export default function AppWidgetSummaryNoIcon({ title, text, isText, color = 'primary', sx, ...other }) {
  if (text === null || text === "" || text === "-") {
    return null;
  }
  return (
    <Card
      sx={{
        py: 3,
        boxShadow: 0,
        textAlign: 'center',
        color: (theme) => theme.palette[color].darker,
        bgcolor: (theme) => theme.palette[color].lighter,
        height: 120,
        ...sx,
      }}
      {...other}
    >

      <Typography variant="h3">
        {text}
      </Typography>

      <Typography variant="subtitle2" sx={{ opacity: 0.72 }}>
        {title}
      </Typography>
    </Card>
  );
}
