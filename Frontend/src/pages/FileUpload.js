import React, { useState } from "react";
import { Button, Upload, Space, Typography, Progress } from "antd";
import '../App.css';
import 'antd/dist/antd.css';
import axios from 'axios';
import { FileOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';


export default function App() {
  const [files, setFiles] = useState({});

  const handleFileUpload = ({ file }) => {
    const getFileObject = (progress, estimated) => {
      return {
        name: file.name,
        uid: file.uid,
        progress: progress,
        estimated: estimated || 0  
      }
    }
    axios.post('__SERVER_URI__/fileUpload', file, {
      onUploadProgress: (event) => {
        setFiles(pre => {
          return { ...pre, [file.uid]: getFileObject(event.progress, event.estimated) };
        });
      }
    });
  };

  const getTimeString = (timeInSeconds) => {
    const hours = Math.floor(timeInSeconds / 3600)
    const minutes = Math.floor(timeInSeconds / 60 - hours * 60)
    const seconds = Math.floor(timeInSeconds - minutes * 60 - hours * 3600)
    let timeString = `${seconds} sec`
    if (minutes) {
      timeString = `${minutes} min ${timeString}`
    }
    if (hours) {
      timeString = `${hours} hrs ${timeString}`
    }
    return timeString
  }

  const totalRemainTime = getTimeString(
    Object.values(files).reduce((total, current) => {
      return total + current.estimated
    }, 0)
  );

  return (
    <div className="background" style={{
      width: '100%',
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      marginTop: 0,
      height: "100%"
    }}>
      <Space 
        direction="vertical" style={{
          alignItems: "center",
        }} 
        
      >
        {Object.values(files).length === 0 ? (
          <h1 className="gradient-text">Upload .zip File</h1>
        ) : (
          <Typography.Text 
              style={{ 
                fontSize: '24px',
                fontWeight: 'bold',
                color: '#999999'
              }}
            >
              Total Remaining Time: {totalRemainTime} 
            </Typography.Text>
        )
        } 
        {Object.values(files).length === 0 && (
        <Upload.Dragger
          multiple
          showUploadList={false}
          accept=".zip"
          customRequest={handleFileUpload}
          style={{
            width: 500,
            fontSize: '17px'
          }}
        >
          <div style={{ marginBottom: '10px' }}>Drag files here or</div>
          <Button className="buttonUpload">Click Upload</Button>
        </Upload.Dragger>
      )} 

      <div
        style={{
          width: 500,
          maxHeight: 300,
          overflow: 'auto',
        }}
      >
        {Object.values(files).map((file, index) => {
          return (
            <Space 
              key={file.uid} 
              direction="vertical" 
              style={{ 
                backgroundColor: "#fafafa", 
                borderColor: '#eaeaea',
                  borderWidth: '1px',
                  borderStyle: 'solid',
                width: 500, 
                padding: 8 
              }}
            >
              <Space>
                <FileOutlined />
                <Typography>{file.name}</Typography>
                {file.estimated ? (
                  <Typography.Text type="secondary"> {" "} is being uploaded in {getTimeString(file.estimated)} </Typography.Text>
                ) : (
                  <Typography.Text type="secondary"> {" "} is Uploaded Successfully </Typography.Text>
                )}
              </Space>
              <Progress
                percent={Math.ceil(file.progress * 100)}
                strokeWidth={10}
                strokeColor={{
                  "0%": "#d9d9d9",
                  "100%": "#999999"
                }}
              />
            </Space>
          );
        })}
      </div>
      {Object.values(files).length > 0 && (
        <Link to="/dashboard">
          <Button className="buttonDashboard">See Dashboard</Button>
        </Link>
      )}
      </Space>
    </div>
  );
}
