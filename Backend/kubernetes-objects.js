const train_suppervisorObject = {
    apiVersion: 'batch/v1',
    kind: 'Job',
    metadata: {
        name: 'train_suppervisor'
    },
    spec: {
        affinity: {
            nodeAffinity: {
                preferredDuringSchedulingIgnoredDuringExecution: {
                    nodeSelectorTerms: [
                        {
                            matchExpressions: [
                                {
                                    key: 'train_suppervisor',
                                    operator: 'In',
                                    values: ['yes']
                                }
                            ]
                        }
                    ]
                }
            }
        },
        template: {
            spec: {
                containers: [
                    {
                        name: 'trainsuppervisor',
                        image: 'rafaelxokito/neuralnetnexustrain_suppervisor:latest'
                    }
                ],
                restartPolicy: 'Never'
            }
        },
        backoffLimit: 4,
        nodeSelector: {
            'kubernetes.io/role': 'helper'
        }
    }
};

module.exports = {
    train_suppervisorObject: train_suppervisorObject
};