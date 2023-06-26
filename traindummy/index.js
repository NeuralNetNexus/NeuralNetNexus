const io = require('socket.io-client');
const { randomInt } = require('crypto');

const WSSERVER = process.env.WSSERVER || 'ws://192.168.1.71';

const socket = io(WSSERVER); // change this to your server's URL
const projectId = '6495c3bea4f442a5776a6d9a';

// Connect to the server
socket.on('connect', () => {
  console.log('Connected to the server');

  // Join the project
  socket.emit('joinProject', projectId);

  // Send dummy data every 5 seconds
  setInterval(() => {
    // Random accuracy and loss for metrics
    const randomAcc = randomInt(60, 90);
    const randomLoss = Math.random() + 1;

    // Send projectStatus
    socket.emit('projectStatus', projectId, {
      "status": "started (2-4)",
    });

    // Send splitNumber
    socket.emit('splitNumber', projectId, {
      "n_batch": 3,
    });

    // Send trainingMetrics
    socket.emit('trainingMetrics', projectId, {
      "train_index": randomInt(1, 6),
      "epoch": 50,
      "train_accuracy": randomAcc,
      "val_accuracy": randomAcc - 5,
      "train_loss": randomLoss,
      "val_loss": randomLoss + 0.3,
      "cpu_usage": 35,
      "ram_usage": 35
    });

    // Send aggregatorMetrics
    socket.emit('aggregatorMetrics', projectId, {
      "test_accuracy": randomAcc
    });
  }, 5000);
});

socket.on('connect_error', (error) => {
  console.log('Connection Error: ', error);
});

socket.on('disconnect', () => {
  console.log('Disconnected from the server');
});
