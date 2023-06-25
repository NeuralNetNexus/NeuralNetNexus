const http = require('http');
const { Server } = require('socket.io');

const PORT = process.env.PORT || 3002;
 
const httpServer = http.createServer();
const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});
 
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('joinProject', (data) => {
      socket.join(data.projectId);
  });

  // Live state of the project from a project supervisor
  socket.on('projectState', (data) => {
    console.log('projectState', data.projectId, data.state);
    io.to(data.projectId).emit('projectState', data.state);
  });

  // A number related to the project from a related client
  socket.on('splitNumber', (data) => {
    console.log('splitNumber', data.projectId, data.number);
    io.to(data.projectId).emit('splitNumber', data.number);
  });

  // Several metrics related to a process from a related client
  socket.on('trainingMetrics', (data) => {
    console.log('trainingMetrics', data.projectId, data.metrics);
    io.to(data.projectId).emit('trainingMetrics', data.metrics);
  });

  // Several metrics related to a final process from a related client
  socket.on('aggregatorMetrics', (data) => {
    console.log('aggregatorMetrics', data.projectId, data.metrics);
    io.to(data.projectId).emit('aggregatorMetrics', data.metrics);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

httpServer.listen(PORT, () => {
  console.log(`Socket.IO server running on port ${PORT}`);
});
