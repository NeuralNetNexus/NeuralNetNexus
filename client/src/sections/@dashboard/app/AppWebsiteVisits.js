import PropTypes from 'prop-types';
import ReactApexChart from 'react-apexcharts';
// @mui
import { Card, CardHeader, Box } from '@mui/material';
// components
import { useChart } from '../../../components/chart';

// ----------------------------------------------------------------------

AppWebsiteVisits.propTypes = {
  title: PropTypes.string,
  subheader: PropTypes.string,
  chartData: PropTypes.array.isRequired,
  chartLabels: PropTypes.arrayOf(PropTypes.string).isRequired,
  height: PropTypes.number,
};

export default function AppWebsiteVisits({ title, subheader, chartLabels, chartData, colors, height, ...other }) {
  const chartOptions = useChart({
    plotOptions: { bar: { columnWidth: '20%' } },
    fill: { type: chartData.map((i) => i.fill) },
    labels: chartLabels,
    xaxis: { title: { text: 'Epoch' }, type: 'string' },
    yaxis: { 
      title: { text: title } ,
      labels: {
        formatter: (value) => {
          return value.toFixed(2);
        },
      },
      tickAmount: 3, 
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: (y) => {
          if (typeof y !== 'undefined') {
            return `${y.toFixed(2)}`;
          }
          return y;
        },
      },
    },
    colors: colors, // Update the colors property with the custom color palette
  });

  return (
    <Card {...other}>
      <CardHeader title={title} subheader={subheader} />

      <Box sx={{ px:2, pb: 1}} dir="ltr">
        <ReactApexChart type="line" series={chartData} options={chartOptions} height={height} /> {/* Use the height prop */}
      </Box>
    </Card>
  );
}
