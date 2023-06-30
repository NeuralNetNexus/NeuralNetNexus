import React, { useEffect, useRef } from 'react';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

const LogDisplay = ({ logs }) => {
  const endOfLogsRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      endOfLogsRef.current.scrollIntoView({ behavior: "smooth" });
    }, 0);

    // Cleanup on unmount
    return () => clearTimeout(timer);
  }, [logs]);

  return (
    <Box sx={{ width: '100%', height: 300, overflow: 'auto' }}>
      <List>
        {logs.map((log, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={
                <React.Fragment>
                  <Typography
                    sx={{ display: 'inline' }}
                    component="span"
                    variant="body2"
                    color="text.primary"
                  >
                    {log}
                  </Typography>
                </React.Fragment>
              }
            />
          </ListItem>
        ))}
        <div ref={endOfLogsRef} />
      </List>
    </Box>
  );
};

export default LogDisplay;