// kubernetes-objects.js

const getTrainSupervisorObject = function (projectId) {
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
  
    return trainSuppervisorObject;
  };
  
  module.exports = {
    getTrainSupervisorObject: getTrainSupervisorObject,
  };
  