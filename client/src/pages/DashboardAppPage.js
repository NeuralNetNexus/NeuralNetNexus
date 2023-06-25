import { Helmet } from 'react-helmet-async';
import { faker } from '@faker-js/faker';
//import axios from 'axios';
import React, { useState, useEffect } from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import { io } from 'socket.io-client';

// @mui
import { useTheme } from '@mui/material/styles';
import { Grid, Container, Typography } from '@mui/material';
// components
import Iconify from '../components/iconify';
// sections
import {
  AppTasks,
  AppNewsUpdate,
  AppOrderTimeline,
  AppCurrentVisits,
  AppWebsiteVisits,
  AppTrafficBySite,
  AppWidgetSummary,
  AppCurrentSubject,
  AppConversionRates,
} from '../sections/@dashboard/app';

// ----------------------------------------------------------------------

export default function DashboardAppPage() {
  /*const theme = useTheme();
  const [modelSize, setModelSize] = useState("0");
  const [currentState, setCurrentState] = useState(false);
  const [n_splits, setNSplit] = useState("0");
  const [accuracies, setAccuracies] = useState("0%");
  const [accuracy, setAccuracy] = useState(0);
  const [loss, setLoss] = useState(0);
  const [epoch, setEpoch] = useState("0");
  const [expanded, setExpanded] = useState(false);
  const [items, setItems] = useState({});
  const [jsonArray, setItemData] = useState([]);*/

  const theme = useTheme();
  const [modelSize, setTrainAccuracy] = useState('');
  const [currentState, setCurrentState] = useState('');
  const [nSplit, setNSplit] = useState('');
  const [trainingData, setTrainingData] = useState([]);
  const [expanded, setExpanded] = useState(false);



  //const items = ['Item 1', 'Item 2', 'Item 3', 'Item 4'];

  
  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };
  

  useEffect(() => {
    const socket = io('ws://192.168.1.71');
 
    socket.on('connect', () => {
      console.log('Connected to the server');
    });

    socket.on('aggregatorMetrics', (values) => {
      setTrainAccuracy(values.test_accuracy);
      console.log('ola');
    });

    socket.on('projectStatus', (values) => {
      setCurrentState(values.status);
      console.log('ola1');
    });

    socket.on('splitNumber', (values) => {
      setNSplit(values.n_batch);
      console.log('ola2');
    });

    socket.on('trainingMetrics', (values) => {
      console.log('ola3');
      // Find the index in the existing array
      const existingIndex = trainingData.findIndex((data) => data.train_index === values.train_index);

      // Create a new object with the updated values
      const updatedData = {
        train_index: values.train_index,
        epoch: values.epoch,
        train_accuracy: values.train_accuracy,
        val_accuracy: values.val_accuracy,
        train_loss: values.train_loss,
        val_loss: values.val_loss,
        cpu_usage: values.cpu_usage,
        ram_usage: values.ram_usage,
      };

      // Update the state based on the existing index
      if (existingIndex !== -1) {
        setTrainingData((prevData) => {
          const updatedArray = [...prevData];
          updatedArray[existingIndex] = updatedData;
          return updatedArray;
        });
      } else {
        // Insert the new object into the array
        setTrainingData((prevData) => [...prevData, updatedData]);
      }
    });

    // Cleanup the websocket connection on component unmount
    return () => {
      socket.disconnect();
    };
    //fetchData();
  }, []);

  /*const fetchData = async () => {


    await axios.get('http://localhost:3001/modelSize')
      .then(response => {
        const values = response.data;
        setModelSize(values.modelSize);
        setCurrentState(values.state);
        setNSplit(values.n_splits)
        setAccuracies(values.accuracies)
        setAccuracy(values.accuracy)
        setLoss(values.loss)
        setEpoch(values.epoch)
        setItems(values)
        jsonArray.push(values)
        console.log(jsonArray[0])
        setItemData(jsonArray)
      })
      .catch(error => {
        console.error(error);
      });
  };*/

  return (
    <>
      <Helmet>
        <title> Dashboard | Neural Net Nexus </title>
      </Helmet>

      <Container maxWidth="xl">
        <Typography variant="h2" sx={{ mb: 5 }}>
          Neural Net Nexus
        </Typography>

        <Grid container spacing={3}>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Status" text={currentState ? 'Active' : 'Inactive'} color={currentState ? 'success' : 'error'} icon={'ant-design:apple-filled'} />
          </Grid>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Model Size" text={modelSize.toString()} icon={'ant-design:android-filled'} />
          </Grid>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Number of Splits" text={nSplit.toString()} color="warning" icon={'ant-design:windows-filled'} />
          </Grid>


          <Typography variant="h4" sx={{ mb: 1, paddingTop: "50px", paddingLeft: "30px"}}>
            Training Pods
          </Typography>

          {trainingData.map((item, index) => (
            <Grid item xs={12} sm={12} md={12} key={index}>
              <Accordion expanded={expanded === `panel${index}`} onChange={handleChange(`panel${index}`)}>
                <AccordionSummary
                  expandIcon={"↑"}
                  aria-controls={`panel${index}bh-content`}
                  id={`panel${index}bh-header`}
                >
                  <Typography sx={{ width: '100%', flexShrink: 0 }}>
                    Train_pod_name
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={4}>
                    <Grid item xs={12} sm={6} md={3}>
                      <AppWidgetSummary title="Train Index" text={item.cpu_usage.toString()} icon={'ant-design:android-filled'} />
                    </Grid>

                    <Grid item xs={12} sm={6} md={3}>
                      <AppWidgetSummary title="Epoch" text={item.epoch.toString()} color="success" icon={'ant-design:android-filled'} />
                    </Grid>

                    <Grid item xs={12} sm={6} md={3}>
                      <AppWidgetSummary title="CPU Usage" text={item.cpu_usage.toString()} color="warning" icon={'ant-design:android-filled'} />
                    </Grid>

                    <Grid item xs={12} sm={6} md={3}>
                      <AppWidgetSummary title="RAM Usage" text={item.ram_usage.toString()} color="error" icon={'ant-design:android-filled'} />
                    </Grid>
                  </Grid>
                  <Grid container spacing={4} sx={{ paddingBottom: '16px', paddingTop: '16px'}}>
                    <Grid item xs={12} md={6} lg={6}>
                      <AppWebsiteVisits 
                        title="Training Accuracy"
                        chartLabels={item.epoch}
                        chartData={[
                          {
                            name: 'Training Accuracy',
                            type: 'area',
                            fill: 'gradient',
                            data: item.data.concat(item.train_accuracy).join(", ")
                          },
                        ]}
                        height={150}
                      />
                    </Grid>
                    <Grid item xs={12} md={6} lg={6}>
                      <AppWebsiteVisits
                        title="Validation Accuracy"
                        chartLabels={item.epoch}
                        chartData={[
                          {
                            name: 'Validation Accuracy',
                            type: 'area',
                            fill: 'gradient',
                            data: item.data.concat(item.val_accuracy).join(", ")
                          },
                        ]}
                        colors={['red']}
                        height={150}
                      />
                    </Grid>
                  </Grid>
                  <Grid container spacing={4} >
                    <Grid item xs={12} md={6} lg={6}>
                      <AppWebsiteVisits
                        title="Training Loss"
                        chartLabels={item.epoch}
                        chartData={[
                          {
                            name: 'Training Loss',
                            type: 'area',
                            fill: 'gradient',
                            data: item.data.concat(item.train_loss).join(", ")
                          },
                        ]}
                        colors={['orange']}
                        height={150}
                      />
                    </Grid>
                    <Grid item xs={12} md={6} lg={6}>
                      <AppWebsiteVisits
                        title="Validation Loss"
                        chartLabels={item.epoch}
                        chartData={[
                          {
                            name: 'Validation Loss',
                            type: 'area',
                            fill: 'gradient',
                            data: item.data.concat(item.val_loss).join(", ")
                          },
                        ]}
                        colors={['green']}
                        height={150}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          ))}

          <Grid item xs={12} md={6} lg={4}>
            <AppCurrentVisits
              title="Current Visits"
              chartData={[
                { label: 'America', value: 4344 },
                { label: 'Asia', value: 5435 },
                { label: 'Europe', value: 1443 },
                { label: 'Africa', value: 4443 },
              ]}
              chartColors={[
                theme.palette.primary.main,
                theme.palette.info.main,
                theme.palette.warning.main,
                theme.palette.error.main,
              ]}
            />
          </Grid>

          <Grid item xs={12} md={6} lg={8}>
            <AppConversionRates
              title="Conversion Rates"
              subheader="(+43%) than last year"
              chartData={[
                { label: 'Italy', value: 400 },
                { label: 'Japan', value: 430 },
                { label: 'China', value: 448 },
                { label: 'Canada', value: 470 },
                { label: 'France', value: 540 },
                { label: 'Germany', value: 580 },
                { label: 'South Korea', value: 690 },
                { label: 'Netherlands', value: 1100 },
                { label: 'United States', value: 1200 },
                { label: 'United Kingdom', value: 1380 },
              ]}
            />
          </Grid>

          <Grid item xs={12} md={6} lg={4}>
            <AppCurrentSubject
              title="Current Subject"
              chartLabels={['English', 'History', 'Physics', 'Geography', 'Chinese', 'Math']}
              chartData={[
                { name: 'Series 1', data: [80, 50, 30, 40, 100, 20] },
                { name: 'Series 2', data: [20, 30, 40, 80, 20, 80] },
                { name: 'Series 3', data: [44, 76, 78, 13, 43, 10] },
              ]}
              chartColors={[...Array(6)].map(() => theme.palette.text.secondary)}
            />
          </Grid>

          <Grid item xs={12} md={6} lg={8}>
            <AppNewsUpdate
              title="News Update"
              list={[...Array(5)].map((_, index) => ({
                id: faker.datatype.uuid(),
                title: faker.name.jobTitle(),
                description: faker.name.jobTitle(),
                image: `/assets/images/covers/cover_${index + 1}.jpg`,
                postedAt: faker.date.recent(),
              }))}
            />
          </Grid>

          <Grid item xs={12} md={6} lg={4}>
            <AppOrderTimeline
              title="Order Timeline"
              list={[...Array(5)].map((_, index) => ({
                id: faker.datatype.uuid(),
                title: [
                  '1983, orders, $4220',
                  '12 Invoices have been paid',
                  'Order #37745 from September',
                  'New order placed #XF-2356',
                  'New order placed #XF-2346',
                ][index],
                type: `order${index + 1}`,
                time: faker.date.past(),
              }))}
            />
          </Grid>

          <Grid item xs={12} md={6} lg={4}>
            <AppTrafficBySite
              title="Traffic by Site"
              list={[
                {
                  name: 'FaceBook',
                  value: 323234,
                  icon: <Iconify icon={'eva:facebook-fill'} color="#1877F2" width={32} />,
                },
                {
                  name: 'Google',
                  value: 341212,
                  icon: <Iconify icon={'eva:google-fill'} color="#DF3E30" width={32} />,
                },
                {
                  name: 'Linkedin',
                  value: 411213,
                  icon: <Iconify icon={'eva:linkedin-fill'} color="#006097" width={32} />,
                },
                {
                  name: 'Twitter',
                  value: 443232,
                  icon: <Iconify icon={'eva:twitter-fill'} color="#1C9CEA" width={32} />,
                },
              ]}
            />
          </Grid>

          <Grid item xs={12} md={6} lg={8}>
            <AppTasks
              title="Tasks"
              list={[
                { id: '1', label: 'Create FireStone Logo' },
                { id: '2', label: 'Add SCSS and JS files if required' },
                { id: '3', label: 'Stakeholder Meeting' },
                { id: '4', label: 'Scoping & Estimations' },
                { id: '5', label: 'Sprint Showcase' },
              ]}
            />
          </Grid>
        </Grid>
      </Container>
    </>
  );
}
