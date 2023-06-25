import { Helmet } from 'react-helmet-async';
import { faker } from '@faker-js/faker';
//import axios from 'axios';
import React, { useState, useEffect } from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import { io } from 'socket.io-client';
import { useParams } from 'react-router-dom';

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
  const [modelSize, setTrainAccuracy] = useState('-');
  const [currentState, setCurrentState] = useState('-');
  const [nSplit, setNSplit] = useState(0);
  const [trainingData, setTrainingData] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const [graphData, setGraphData] = useState([]);
  
  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };
  
  let [train_accuracy, setTrainAccurary] = useState([]);

  useEffect(() => {
    const socket = io('ws://192.168.1.71');
 
    socket.on('connect', () => {
      console.log('Connected to the server');
      socket.emit("joinProject", "meicm123")

    });

    socket.on('aggregatorMetrics', (values) => {
      setTrainAccuracy(values.test_accuracy);
    });

    socket.on('projectStatus', (values) => {
      setCurrentState(values.status);
    });

    socket.on('splitNumber', (values) => {
      setNSplit(values.n_batch);
    });

    socket.on('trainingMetrics', (values) => {
      setGraphData((prevGraphData) => {
        // Create a copy of the previous graph data
        const updatedGraphData = [...prevGraphData];
    
        // Find the index of the source in the graph data
        const sourceIndex = updatedGraphData.findIndex(
          (data) => data.source === values.train_index
        );
    
        // Create a new data object for the current source
        const newData = {
          source: values.train_index,
          trainAccuracy: [],
          valAccuracy: [],
          trainLoss: [],
          valLoss: [],
        };
    
        // If the source already exists in the graph data, update its metrics
        if (sourceIndex !== -1) {
          updatedGraphData[sourceIndex] = {
            ...updatedGraphData[sourceIndex],
            ...newData,
          };
        } else {
          // If the source is new, add it to the graph data
          updatedGraphData.push(newData);
        }
    
        // Push the metrics to the corresponding arrays for the current source
        const sourceData = updatedGraphData.find(
          (data) => data.source === values.train_index
        );
        sourceData.trainAccuracy.push((prevGraphData) => [...prevGraphData, values.train_accuracy]);
        sourceData.valAccuracy.push(values.val_accuracy);
        sourceData.trainLoss.push(values.train_loss);
        sourceData.valLoss.push(values.val_loss);
    
        return updatedGraphData;
      });
    });    

    // Cleanup the websocket connection on component unmount
    return () => {
      socket.disconnect();
    };
    //fetchData();
  }, []);

  const { id } = useParams();

  return (
    <>
      <Helmet>
        <title> Dashboard | Neural Net Nexus </title>
      </Helmet>

      <Container maxWidth="xl">
        <Typography variant="h3" sx={{ mb: 5 }}>
          Project nº {id}
        </Typography>

        <Grid container spacing={3}>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Status" text={currentState ? 'Active' : 'Inactive'} color={currentState ? 'success' : 'error'} icon={'ant-design:apple-filled'} />
          </Grid>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Model Size" text={modelSize} icon={'ant-design:android-filled'} />
          </Grid>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Number of Splits" text={nSplit} color="warning" icon={'ant-design:windows-filled'} />
          </Grid>


          <Typography variant="h4" sx={{ mb: 1, paddingTop: "50px", paddingLeft: "30px"}}>
            Training Pods
          </Typography>

          {graphData.map((item, index) => (
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
                    <AppWidgetSummary title="Train Index" text={item.source} icon={'ant-design:android-filled'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <AppWidgetSummary title="Epoch" text={item.epoch} color="success" icon={'ant-design:android-filled'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <AppWidgetSummary title="CPU Usage" text={item.cpu_usage} color="warning" icon={'ant-design:android-filled'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <AppWidgetSummary title="RAM Usage" text={item.ram_usage} color="error" icon={'ant-design:android-filled'} />
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
                          data: item.trainAccuracy, // Update to item.trainAccuracy
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
                          data: item.valAccuracy, // Update to item.valAccuracy
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
                          data: item.trainLoss, // Update to item.trainLoss
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
                          data: item.valLoss, // Update to item.valLoss
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
        </Grid>
      </Container>
    </>
  );
}
