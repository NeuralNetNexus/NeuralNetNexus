// kubernetes-objects.js

const getTrainSupervisorObject = function (projectId, model) {
    const trainSupervisorObject = {
      apiVersion: 'batch/v1',
      kind: 'Job',
      metadata: {
        name: `train-supervisor-${projectId}`,
      },
      spec: {
          affinity: {
            nodeAffinity: {
                preferredDuringSchedulingIgnoredDuringExecution: {
                    nodeSelectorTerms: [
                        {
                            matchExpressions: [
                                {
                                    key: 'helper',
                                    operator: 'In',
                                    values: ['yes']
                                }
                            ]
                        }
                    ]
                }
            }
        },
        ttlSecondsAfterFinished: 10,
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
                  {
                    name: 'MODEL',
                    value: model,
                  },
                ],
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
    return trainSupervisorObject;
  };
  
  module.exports = {
    getTrainSupervisorObject: getTrainSupervisorObject,
  };
  