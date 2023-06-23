import { Helmet } from 'react-helmet-async';
import React, { useState } from 'react';
import axios from 'axios';
import { Container, Typography, Box, Select, MenuItem, InputLabel, Button, FormControl, TextField, LinearProgress } from '@mui/material';

const ProjectPage = () => {
    const [projectName, setProjectName] = useState('');
    const [selectedNet, setSelectedNet] = useState('');
    const [dataset, setDataset] = useState(null);
    const [loading, setLoading] = useState(false);

    const neuralNets = ['VGG-16', 'ResNet18', 'ResNet50', 'EfficiencyNet V2', 'AlexNet', 'ViT'];

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
        formData.append('text', "ola");
        formData.append('projectName', projectName);
        formData.append('network', selectedNet);
        formData.append('dataset', dataset);
          
        console.log(dataset)
        try {
            const response = await axios.post('http://localhost:3001/api/upload', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });
            console.log(response.data);
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
