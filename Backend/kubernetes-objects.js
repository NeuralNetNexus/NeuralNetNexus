// kubernetes-objects.js

const trainSuppervisorObject = {
    apiVersion: 'batch/v1',
    kind: 'Job',
    metadata: {
      name: 'train-suppervisor',
    },
    spec: {
      affinity: {
        nodeAffinity: {
          preferredDuringSchedulingIgnoredDuringExecution: [
            {
              weight: 1,
              preference: {
                matchExpressions: [
                  {
                    key: 'train-suppervisor',
                    operator: 'In',
                    values: ['yes'],
                  },
                ],
              },
            },
          ],
        },
      },
      template: {
        spec: {
          containers: [
            {
              name: 'trainsuppervisor',
              image: 'rafaelxokito/neuralnetnexustrain_suppervisor:latest',
            },
          ],
          restartPolicy: 'Never',
        },
      },
      backoffLimit: 4,
      nodeSelector: {
        'kubernetes.io/role': 'helper',
      },
    },
  };
  
  module.exports = {
    getTrainSupervisorObject: function (projectId) {
      // Customize and return the Kubernetes Job manifest based on the projectId
      // In this example, we are using the same object for all projects
      return trainSuppervisorObject;
    },
  };
  