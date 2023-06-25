import { Helmet } from 'react-helmet-async';
import axios from 'axios';
import React, { useState, useEffect } from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import { io } from 'socket.io-client';
import { useParams } from 'react-router-dom';

// @mui
import { Grid, Container, Typography } from '@mui/material';

// sections
import {
  AppWebsiteVisits,
  AppWidgetSummary,
} from '../sections/@dashboard/app';

// ----------------------------------------------------------------------

export default function DashboardAppPage() {
  const { id } = useParams();
  const [trainAccuracy, setTrainAccuracy] = useState(null);
  const [currentState, setCurrentState] = useState('-');
  const [nSplit, setNSplit] = useState('-');
  const [expanded, setExpanded] = useState(false);
  const [graphData, setGraphData] = useState([]);
  
  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };
  
  useEffect(async () => {
    
    try {
      const response = await axios.get(`/api/projects/${id}`);
      const project = response.data
      console.log(project);

      setTrainAccuracy(project.aggregated_accuracy)
      setCurrentState(project.state)
      setNSplit(project.n_splits)
      setGraphData(() => {

        let data = []
        for (let i = 0; i < project.n_splits; i++) {
          const size = project.splits[i].train_accuracies.length
          let epochs = Array.from({ length: size }, (_, index) => index + 1);

          let obj = {
            epoch: epochs,
            trainAccuracy: project.splits[i].train_accuracies,
            valAccuracy: project.splits[i].train_accuracies,
            trainLoss: project.splits[i].train_accuracies,
            valLoss: project.splits[i].train_accuracies
          }
          data.push(obj)
        }

        return data
      });

    } catch (err) {
      console.error(err);
    }

    const socket = io(window.location.host);
 
    socket.on('connect', () => {
      console.log('Connected to the server');
      socket.emit("joinProject", id)

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
    
        // If the source already exists in the graph data, update its metrics
        if (sourceIndex !== -1) {
          const sourceData = updatedGraphData[sourceIndex];
          sourceData.epoch.push(sourceData.trainAccuracy.length + 1);
          sourceData.trainAccuracy.push(values.train_accuracy);
          sourceData.valAccuracy.push(values.val_accuracy);
          sourceData.trainLoss.push(values.train_loss.toFixed(2));
          sourceData.valLoss.push(values.val_loss.toFixed(2));
        } else {
          // If the source is new, create a new data object with arrays
          const newData = {
            source: updatedGraphData.length + 1,
            cpu_usage: values.cpu_usage,
            ram_usage: values.ram_usage,
            trainAccuracy: [values.train_accuracy],
            valAccuracy: [values.val_accuracy],
            trainLoss: [values.train_loss.toFixed(2)],
            valLoss: [values.val_loss.toFixed(2)],
            epoch: [1],
          };
          updatedGraphData.push(newData);
        }
    
        return updatedGraphData;
      });
    });
        
    // Cleanup the websocket connection on component unmount
    return () => {
      socket.disconnect();
    };
    //fetchData();
  }, [id]);

  return (
    <>
      <Helmet>
        <title> Dashboard | Neural Net Nexus </title>
      </Helmet>

      <Container maxWidth="xl">
        <Typography variant="h3" sx={{ mb: 5 }}>
          Project Nº {id}
        </Typography>

        <Grid container spacing={3}>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Status" text={currentState} color={currentState ? 'success' : 'error'} icon={'ant-design:apple-filled'} />
          </Grid>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Test Accuracy" text={trainAccuracy ? trainAccuracy + "%" : "-"} icon={'ant-design:android-filled'} />
          </Grid>

          <Grid item xs={12} sm={4} md={4}>
            <AppWidgetSummary title="Number of Splits" text={nSplit} color="warning" icon={'ant-design:windows-filled'} />
          </Grid>

          { graphData.length > 0 ?
            <Typography variant="h4" sx={{ mb: 1, paddingTop: "50px", paddingLeft: "30px"}}>
              Training Pods
            </Typography>
            : null
          }

          {graphData.map((item, index) => (
          <Grid item xs={12} sm={12} md={12} key={index}>
            <Accordion expanded={expanded === `panel${index}`} onChange={handleChange(`panel${index}`)}>
              <AccordionSummary
                expandIcon={"↑"}
                aria-controls={`panel${index}bh-content`}
                id={`panel${index}bh-header`}
              >
                <Typography sx={{ width: '100%', flexShrink: 0 }}>
                  Pod #{item.source}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={4}>
                  <Grid item xs={12} sm={6} md={4}>
                    <AppWidgetSummary title="Epoch" text={String(item.epoch[-1] || 1)} color="success" icon={'ant-design:android-filled'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={4}>
                    <AppWidgetSummary title="CPU Usage" text={item.cpu_usage} color="warning" icon={'ant-design:android-filled'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={4}>
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
