// component
import SvgColor from '../../../components/svg-color';

// ----------------------------------------------------------------------

const icon = (name) => <SvgColor src={`/assets/icons/navbar/${name}.svg`} sx={{ width: 1, height: 1 }} />;

const navConfig = [
  // {
  //   title: 'dashboard',
  //   path: '/dashboard/app',
  //   icon: icon('ic_account_tree'),
  // },
  {
    title: 'projects',
    path: '/dashboard/projects',
    icon: icon('ic_account_tree'),
  },
];

export default navConfig;
