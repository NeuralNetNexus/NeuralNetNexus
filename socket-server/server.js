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
      console.log('joinProject', data.projectId);
      socket.join(data.projectId);
  });

  // Live state of the project from a project supervisor
  socket.on('projectState', (data) => {
    console.log('projectState', data);
    io.to(data.projectId).emit('projectState', data);
  });

  // A number related to the project from a related client
  socket.on('splitNumber', (data) => {
    console.log('splitNumber', data);
    io.to(data.projectId).emit('splitNumber', data);
  });

  // Several metrics related to a process from a related client
  socket.on('trainingMetrics', (data) => {
    console.log('trainingMetrics', data);
    io.to(data.projectId).emit('trainingMetrics', data);
  });

  // Several metrics related to a final process from a related client
  socket.on('aggregatorMetrics', (data) => {
    console.log('aggregatorMetrics', data);
    io.to(data.projectId).emit('aggregatorMetrics', data);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

httpServer.listen(PORT, () => {
  console.log(`Socket.IO server running on port ${PORT}`);
});
