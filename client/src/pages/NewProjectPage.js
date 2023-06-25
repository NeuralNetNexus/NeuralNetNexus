import { Helmet } from 'react-helmet-async';
import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, Box, Select, MenuItem, InputLabel, Button, FormControl, TextField, LinearProgress } from '@mui/material';
import io from 'socket.io-client';
const socket = io(window.location.host);

const ProjectPage = () => {
    const navigate = useNavigate();
    const [projectName, setProjectName] = useState('');
    const [selectedNet, setSelectedNet] = useState('');
    const [dataset, setDataset] = useState(null);
    const [loading, setLoading] = useState(false);

    const neuralNets = ['VGG-16', 'ResNet18', 'EfficientNet V2S', 'SqueezeNet'];

    const handleNetChange = (e) => {
        setSelectedNet(e.target.value);
    };

    const handleFileChange = (e) => {
        setDataset(e.target.files[0]);
    };

    const handleNameChange = (e) => {
        setProjectName(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        
        const formData = new FormData();
        formData.append('projectName', projectName);
        formData.append('model', selectedNet);
        formData.append('dataset', dataset);
          
        console.log(dataset)
        try {
            const response = await axios.post('/api/upload', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });
            console.log(response.data);
            const id = response.data.projectId
            socket.emit('joinProject', id);
            navigate(`/dashboard/projects/${id}`);
          } catch (err) {
            console.error(err);
          } finally {
            setLoading(false);
          }
    };

    return (
      <>
        <Helmet>
          <title> My New Project | Neural Net Nexus </title>
        </Helmet>
        <Container>
            <Box sx={{ padding: '2em' }}>
                <Typography variant="h3" gutterBottom>My New Project</Typography>
                
                <Box my={2}>
                    <TextField
                        label="Project Name"
                        value={projectName}
                        onChange={handleNameChange}
                        fullWidth
                    />
                </Box>

                <FormControl fullWidth variant="filled">
                    <InputLabel id="net-label">Select Neural Network</InputLabel>
                    <Select
                        labelId="net-label"
                        value={selectedNet}
                        onChange={handleNetChange}
                    >
                        {neuralNets.map(net => (
                            <MenuItem key={net} value={net}>{net}</MenuItem>
                        ))}
                    </Select>
                </FormControl>

                <Box my={2}>
                    <InputLabel htmlFor="dataset-upload">Upload Dataset</InputLabel>
                    <input
                        id="dataset-upload"
                        type="file"
                        onChange={handleFileChange}
                        accept=".zip"
                        hidden
                    />
                    <label htmlFor="dataset-upload">
                        <Button variant="contained" component="span">
                        Upload
                        </Button>
                    </label>
                </Box>
                
                <Button variant="contained" color="primary" onClick={handleSubmit} disabled={loading || !selectedNet || !dataset || !projectName}>
                    {loading ? <LinearProgress /> : 'Submit'}
                </Button>
            </Box>
        </Container>
      </>
    );
};

export default ProjectPage;
