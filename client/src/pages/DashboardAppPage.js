import { Helmet } from 'react-helmet-async';
import axios from 'axios';
import React, { useState, useEffect } from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import { io } from 'socket.io-client';
import { useParams } from 'react-router-dom';
import Scrollbar from '../components/scrollbar';
import Iconify from '../components/iconify';
// sections
import {
  AppWebsiteVisits,
  AppWidgetSummary,
  AppWidgetSummaryNoIcon,
  LogDisplay,
} from '../sections/@dashboard/app';

// sections
import { ProjectListHead, ProjectListToolbar } from '../sections/@dashboard/project';


import {
  Grid,
  Table,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
  Container,
  Typography,
  TableContainer,
  TablePagination,
  Checkbox
} from '@mui/material';
// ----------------------------------------------------------------------

export default function DashboardAppPage() {
  const { id } = useParams();
  const [currentState, setCurrentState] = useState("Pending");
  const [nSplit, setNSplit] = useState('-');
  const [name, setName] = useState('-');
  const [logs, setLogs] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const [expandedLogs, setExpandedLogs] = useState(false);
  const [graphData, setGraphData] = useState([]);

  const [accuracy_avg, setAvgAccuracy] = useState('-');
  const [loss_avg, setAvgLoss] = useState('-');
  const [precision, setPrecision] = useState('-');
  const [recall, setRecall] = useState('-');
  const [scoreF1, setF1Score] = useState('-');
  
  const [files, setFiles] = useState([`confusion_matrix_${id}.png`, `model_${id}.pth`]);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [page, setPage] = useState(0);

  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };
  const handleChangeLogs = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setPage(0);
    setRowsPerPage(parseInt(event.target.value, 10));
  };


  useEffect(() => {
    
    const fetchData = async () => {
      try {
        const response = await axios.get(`/api/projects/${id}`);
        const project = response.data.project;
        console.log(project);

        setCurrentState(project.state);
        setNSplit(project.n_splits || '-');
        setName(project.name);
        
        setLogs((previousLogs) => [...previousLogs, ...project.logs]);

        setAvgAccuracy(project.aggregator.accuracy.toFixed(2));
        setAvgLoss(project.aggregator.loss.toFixed(2));
        setPrecision(project.aggregator.precision.toFixed(2));
        setRecall(project.aggregator.recall.toFixed(2));
        setF1Score(project.aggregator.f1_score.toFixed(2));

        setGraphData(() => {
          let data = [];
          for (let i = 0; i < project.n_splits; i++) {
            const size = project.splits[i].train_accuracies.length;
            let epochs = Array.from({ length: size }, (_, index) => index + 1);
  
            let obj = {
              source: i+1,
              epoch: epochs,
              cpu_usage: project.splits[i].cpu_usage,
              ram_usage: project.splits[i].ram_usage,
              trainAccuracy: project.splits[i].train_accuracies.map((value) => value.toFixed(2)),
              valAccuracy: project.splits[i].val_accuracies.map((value) => value.toFixed(2)),
              trainLoss: project.splits[i].train_losses.map((value) => value.toFixed(2)),
              valLoss: project.splits[i].val_losses.map((value) => value.toFixed(2)),
              logs: project.splits[i].logs,
            };
            data.push(obj);
          }
  
          return data;
        });
      
        const socket = io(window.location.host);
        socket.on('connect', () => {
          console.log('Connected to the server');
          socket.emit('joinProject', {'projectId': id});
        });
  
        socket.on('projectState', (values) => {
          setCurrentState(values.state);
        });
  
        socket.on('splitNumber', (values) => {
          setNSplit(values.n_batch);
          setGraphData(() => {
            let data = [];
            for (let i = 0; i < values.n_batch; i++) {
              let obj = {
                source: i+1,
                epoch: [],
                trainAccuracy: [],
                valAccuracy: [],
                trainLoss: [],
                valLoss: [],
                logs: [],
              };
              data.push(obj);
            }
    
            return data;
          });
        });

        socket.on('projectLogs', (values) => {
          console.log('trainingLogs ', values);
          setLogs((prevLogs) => [...prevLogs, values.logs]);
        });

        socket.on('trainingSplitLogs', (values) => {
          console.log('trainingSplitLogs ', values);
          setGraphData((prevGraphData) => {
            const updatedGraphData = [...prevGraphData];
            const sourceIndex = updatedGraphData.findIndex((data) => data.source === values.jobIndex);
            const sourceData = updatedGraphData[sourceIndex];
            sourceData.logs.push(values.logs);
            return updatedGraphData;
          });
        });

        socket.on('aggregatorMetrics', (values) => {
          setAvgLoss(values.loss.toFixed(2));
          setAvgAccuracy(values.accuracy.toFixed(2));
          setPrecision(values.precision.toFixed(2));
          setRecall(values.recall.toFixed(2));
          setF1Score(values.f1_score.toFixed(2));
        });
  
        socket.on('trainingMetrics', (values) => {
          setGraphData((prevGraphData) => {
            const updatedGraphData = [...prevGraphData];
            const sourceIndex = updatedGraphData.findIndex((data) => data.source === values.train_index);
  
            if (sourceIndex !== -1) {
              const sourceData = updatedGraphData[sourceIndex];

              sourceData.cpu_usage = values.cpu_usage;
              sourceData.ram_usage = values.ram_usage;
              sourceData.epoch.push(sourceData.trainAccuracy.length + 1);
              sourceData.trainAccuracy.push(values.train_accuracy.toFixed(2));
              sourceData.valAccuracy.push(values.val_accuracy.toFixed(2));
              sourceData.trainLoss.push(values.train_loss.toFixed(2));
              sourceData.valLoss.push(values.val_loss.toFixed(2));
            } else {
              const newData = {
                source: updatedGraphData.length + 1,
                cpu_usage: values.cpu_usage,
                ram_usage: values.ram_usage,
                trainAccuracy: [values.train_accuracy.toFixed(2)],
                valAccuracy: [values.val_accuracy.toFixed(2)],
                trainLoss: [values.train_loss.toFixed(2)],
                valLoss: [values.val_loss.toFixed(2)],
                epoch: [1],
              };
              updatedGraphData.push(newData);
            }
            return updatedGraphData;
          });
        });
  
        return () => {
          socket.disconnect();
        };
      } catch (err) {
        console.error(err);
      }
    };
  
    fetchData();
  }, []);
  const [selected, setSelected] = useState([]);
  return (
    <>
      <Helmet>
        <title> Dashboard | Neural Net Nexus </title>
      </Helmet>

      <Container maxWidth="xl">
        <Typography variant="h3" sx={{ mb: 5 }}>
          Project {name}
        </Typography>

        <Grid container spacing={4} paddingBottom={10}>

          <Grid item xs={12} sm={6} md={nSplit !== "-" ? 6 : 12}>
            <AppWidgetSummary title="Status" text={currentState} color={currentState ? 'warning' : 'error'} icon={'ant-design:info-outlined'} />
          </Grid>
          
          <Grid item xs={12} sm={6} md={nSplit !== "-" ? 6 : 0}>
            <AppWidgetSummary title="Number of Splits" text={nSplit} color="info" icon={'ant-design:split-cells-outlined'} />
          </Grid>

          { accuracy_avg > 0 ?
          <>
          <Grid item xs={12} sm={6} md={2}>
            <AppWidgetSummaryNoIcon title="Accuracy" text={accuracy_avg} color="secondary" />
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <AppWidgetSummaryNoIcon title="Loss" text={loss_avg} color="secondary" />
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <AppWidgetSummaryNoIcon title="Precision" text={precision} color="secondary"/>
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <AppWidgetSummaryNoIcon title="Recall" text={recall} color="secondary" />
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <AppWidgetSummaryNoIcon title="F1 Score" text={scoreF1} color="secondary"  />
          </Grid>
          </>
          : null}

          { logs.length > 0 ?
          <Grid item md={12}>
          <Accordion>
              <AccordionSummary
                expandIcon={"↑"}
              >
                <Typography variant="h5" sx={{ paddingTop: "30px", paddingLeft: "30px"}}>
                  Logs
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <LogDisplay logs={logs} />
              </AccordionDetails>
            </Accordion>
          </Grid>
          : null}
          
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
                    <AppWidgetSummary title="Epoch" text={String(item.epoch.slice(-1))} color="success" icon={'ant-design:field_ number-outlined'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={4}>
                    <AppWidgetSummary title="CPU Usage" text={item.cpu_usage} color="warning" icon={'ant-design:number-outlined'} />
                  </Grid>
                  <Grid item xs={12} sm={6} md={4}>
                    <AppWidgetSummary title="RAM Usage" text={item.ram_usage} color="error" icon={'ant-design:number-outlined'} />
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
                { logs.length > 0 ?
                <Grid item md={12}>
                <Accordion>
                    <AccordionSummary
                      expandIcon={"↑"}
                    >
                      <Typography variant="h5" sx={{ mb: 1, paddingTop: "50px", paddingLeft: "30px"}}>
                        Trainning Logs
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <LogDisplay logs={item.logs} sx={{ mb: 1, paddingTop: "80px", paddingLeft: "60px"}} />
                    </AccordionDetails>
                  </Accordion>
                </Grid>
                : null}
              </AccordionDetails>
            </Accordion>
          </Grid>
        ))}
        </Grid>

        {precision > 0 ?
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Grid item xs={12} sm={6} md={6} paddingBottom={10} paddingTop={10}>
            <img src={`/api/models/${files[0]}`} alt="Confusion Matrix" />
          </Grid>
        </div>
         : null}

        <Typography variant="h5" sx={{ mb: 2 }}>
          Project Files
        </Typography>

        <Grid item xs={12} sm={6} md={6}>
          <Scrollbar>
            <TableContainer sx={{ minWidth: 800 }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell align="left">File Name</TableCell>
                    <TableCell align="right">Download</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {precision > 0 ?
                  files.map((file) => {
                    const fileName = file.split(".")[0];
                    if(file.includes("png")){
                      return;
                    }
                    return (
                      <TableRow hover key={file} tabIndex={-1}>

                        <TableCell align="left" component="th" scope="row" padding="none">
                          <Typography variant="subtitle2" noWrap>
                            {file}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <a
                            href={`/api/models/${fileName}.pth`}
                            target="_blank"
                          >
                            <Iconify icon={'eva:download-outline'} sx={{ mr: 2 }} />
                          </a>
                        </TableCell>
                      </TableRow>
                    );
                  })
                  : null}
                </TableBody>
              </Table>
            </TableContainer>
          </Scrollbar>
            <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={files.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </Grid>
      </Container>
    </>
  );
}
