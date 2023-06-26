const io = require('socket.io-client');
const { randomInt } = require('crypto');

const WSSERVER = process.env.WSSERVER || 'ws://192.168.1.71';

const socket = io(WSSERVER); // change this to your server's URL
const projectId = '649911265f2b80c4e753b2d8';

// Connect to the server
socket.on('connect', () => {
  console.log('Connected to the server');

  // Join the project
  socket.emit('joinProject', {'projectId': projectId});

  // Send dummy data every 5 seconds
  setInterval(() => {
    // Random accuracy and loss for metrics
    const randomAcc = randomInt(60, 90);
    const randomLoss = Math.random() + 1;

    // Send projectStatus
    socket.emit('projectStatus', {
      'projectId': projectId,
      "state": "started (2-4)",
    });

    // Send splitNumber
    socket.emit('splitNumber', {
      'projectId': projectId,
      "n_batch": 3,
    });

    // Send trainingMetrics
    socket.emit('trainingMetrics', {
      'projectId': projectId,
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
    socket.emit('aggregatorMetrics', {
      'projectId': projectId,
      "accuracy": randomAcc,
      "loss": randomLoss,
      "precision": randomInt(60, 90),
      "recall": randomInt(60, 90),
      "f1score": randomInt(60, 90)
    });
  }, 5000);
});

socket.on('connect_error', (error) => {
  console.log('Connection Error: ', error);
});

socket.on('disconnect', () => {
  console.log('Disconnected from the server');
});
