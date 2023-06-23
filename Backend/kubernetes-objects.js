// kubernetes-objects.js

const getTrainSupervisorObject = function (projectId) {
    const trainSupervisorObject = {
      apiVersion: 'batch/v1',
      kind: 'Job',
      metadata: {
        name: 'train-supervisor',
      },
      spec: {
        template: {
          spec: {
            containers: [
              {
                name: 'trainsupervisor',
                image: 'rafaelxokito/neuralnetnexustrain_suppervisor:latest',
                env: [
                  {
                    name: 'PROJECT_ID',
                    value: projectId,
                  },
                ],
                volumeMounts: [
                  {
                    name: 'datasets-data',
                    mountPath: '/usr/app/datasets',
                  },
                ],
              },
            ],
            restartPolicy: 'Never',
            volumes: [
              {
                name: 'datasets-data',
                persistentVolumeClaim: {
                  claimName: 'pvc-datasets',
                },
              },
            ],
          },
        },
        backoffLimit: 4,
        nodeSelector: {
          'kubernetes.io/role': 'helper',
        },
      },
    };
    return trainSupervisorObject;
  };
  
  module.exports = {
    getTrainSupervisorObject: getTrainSupervisorObject,
  };
  