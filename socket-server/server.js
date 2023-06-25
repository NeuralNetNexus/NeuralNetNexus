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

  socket.on('joinProject', (projectId) => {
      socket.join(projectId);
  });

  // Live state of the project from a project supervisor
  socket.on('projectState', (projectId, state) => {
    console.log('projectState', projectId, state);
    io.to(projectId).emit('projectState', state);
  });

  // A number related to the project from a related client
  socket.on('splitNumber', (projectId, number) => {
    console.log('splitNumber', projectId, number);
    io.to(projectId).emit('splitNumber', number);
  });

  // Several metrics related to a process from a related client
  socket.on('trainingMetrics', (projectId, metrics) => {
    console.log('trainingMetrics', projectId, metrics);
    io.to(projectId).emit('trainingMetrics', metrics);
  });

  // Several metrics related to a final process from a related client
  socket.on('aggregatorMetrics', (projectId, metrics) => {
    console.log('aggregatorMetrics', projectId, metrics);
    io.to(projectId).emit('aggregatorMetrics', metrics);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

httpServer.listen(PORT, () => {
  console.log(`Socket.IO server running on port ${PORT}`);
});
