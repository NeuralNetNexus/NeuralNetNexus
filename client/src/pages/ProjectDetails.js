import { Helmet } from 'react-helmet-async';
import { useParams } from 'react-router-dom';

// @mui
import {
  Stack,
  Container,
  Typography,
} from '@mui/material';

export default function ProjectDetails() {
  const { id } = useParams();
  return (
    <>
      <Helmet>
        <title> Project | Neural Net Nexus </title>
      </Helmet>

      <Container>
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={5}>
          <Typography variant="h4" gutterBottom>
            Project {id}
          </Typography>
        </Stack>       
      </Container>
    </>
  );
}
