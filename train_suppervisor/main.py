from os import getenv
from time import sleep
import requests
import sys
import time

import socketio
import yaml

from kubernetes import client, config

sio = socketio.Client()

project_id = getenv('PROJECT_ID')
model = getenv('MODEL')

def create_job_object(job_name, image_name, env_vars=None, completions=None, parallelism=None):
    # Construct the environment variables for the container
    env = []
    if env_vars is not None:
        for name, value in env_vars.items():
            env.append(client.V1EnvVar(name=name, value=str(value)))
    
    # Define the job's container
    container = client.V1Container(
        name=job_name,
        image=image_name,
        env=env if env else None,
    )

    # Define the job's spec
    if job_name.split('-')[0] != 'split':
        affinity = client.V1Affinity(
            node_affinity=client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                    node_selector_terms=[
                        client.V1NodeSelectorTerm(
                            match_expressions=[
                                client.V1NodeSelectorRequirement(
                                    key="computing",
                                    operator="In",
                                    values=["yessir"]
                                )
                            ]
                        )
                    ]
                )
            )
        )
        # Define the job's template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": job_name.split('-')[0]}),
            spec=client.V1PodSpec(restart_policy="Never", containers=[container], affinity=affinity),
        )
        spec = client.V1JobSpec(
            ttl_seconds_after_finished=10,
            completion_mode="Indexed",
            completions=completions,
            parallelism=parallelism,
            template=template,
            backoff_limit=4,
        )
    else:
        affinity = client.V1Affinity(
            node_affinity=client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                    node_selector_terms=[
                        client.V1NodeSelectorTerm(
                            match_expressions=[
                                client.V1NodeSelectorRequirement(
                                    key="helper",
                                    operator="In",
                                    values=["yessir"]
                                )
                            ]
                        )
                    ]
                )
            )
        )
        # Define the job's template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": job_name.split('-')[0]}),
            spec=client.V1PodSpec(restart_policy="Never", containers=[container], affinity=affinity),
        )
        spec = client.V1JobSpec(
            ttl_seconds_after_finished=10,
            template=template,
            backoff_limit=4,
        )

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=spec,
    )

    return job

def create_job(api_instance, job):
    api_response = api_instance.create_namespaced_job(body=job, namespace="default")
    print("Job created. status='%s'" % str(api_response.status))

def get_job_status(api_instance, job_name):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=job_name,
            namespace="default")
        if api_response.status.succeeded is not None:
            job_completed = True
        sleep(1)

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_incluster_config()
    
    time.sleep(10)

    batch_v1 = client.BatchV1Api()

    print("Waiting for project to be created...")

    sio.connect(f'ws://socket-service', namespaces=['/'])
    sio.emit('joinProject', {'projectId': project_id})
    
    # =================  Split Job  ================= #
    requests.patch(f"http://backend-service/projects/{project_id}/logs", json={"logs": "# =================  Split Job  ================= #"})
    sio.emit('projectLogs', {'projectId': project_id, 'logs': '# =================  Split Job  ================= #'})

    print("Project created. Starting split job...")

    requests.patch(f"http://backend-service/projects/{project_id}/state", json={"state": "[1/4] Splitting Dataset"})
    sio.emit('projectState', {'projectId': project_id, 'state': '[1/4] Splitting Dataset'})

    split_job_name = f"split-job-{project_id}"
    image_name = "rafaelxokito/neuralnetnexussplit:latest"
    env_vars = {"PROJECT_ID": project_id}

    split_job = create_job_object(split_job_name, image_name, env_vars)
    create_job(batch_v1, split_job)
    get_job_status(batch_v1, split_job_name)

    print("Split job finished. Starting training job...")

    # =================  Training Jobs  ================= #
    requests.patch(f"http://backend-service/projects/{project_id}/logs", json={"logs": "# =================  Training Jobs  ================= #"})
    sio.emit('projectLogs', {'projectId': project_id, 'logs': '# =================  Training Jobs  ================= #'})

    requests.patch(f"http://backend-service/projects/{project_id}/state", json={"state": "[2/4] Distributed Training"})
    sio.emit('projectState', {'projectId': project_id, 'state': '[2/4] Distributed Training'})

    train_job_name = f"train-job-{project_id}"
    image_name = "rafaelxokito/neuralnetnexustrain:latest"
    n_splits = requests.get(f"http://backend-service/projects/{project_id}").json()["project"]["n_splits"]

    env_vars = {"PROJECT_ID": project_id, "MODEL": model, "N_SPLITS": n_splits}
    train_job = create_job_object(train_job_name, image_name, env_vars, completions=n_splits, parallelism=n_splits)
    create_job(batch_v1, train_job)
    for _ in range(n_splits):
        get_job_status(batch_v1, train_job_name)

    print("Training job finished. Starting aggregator job...")

    # # =================  Aggregator Job  ================= #
    requests.patch(f"http://backend-service/projects/{project_id}/logs", json={"logs": "# =================  Aggregator Jobs  ================= #"})
    sio.emit('projectLogs', {'projectId': project_id, 'logs': '# =================  Aggregator Jobs  ================= #'})
    
    requests.patch(f"http://backend-service/projects/{project_id}/state", json={"state": "[3/4] Aggregating"})
    sio.emit('projectState', {'projectId': project_id, 'state': '[3/4] Aggregating'})

    aggregator_job_name = f"aggregator-job-{project_id}"
    image_name = "rafaelxokito/neuralnetnexusaggregator:latest"
    aggregator_job = create_job_object(aggregator_job_name, image_name, env_vars)
    create_job(batch_v1, aggregator_job)
    get_job_status(batch_v1, aggregator_job_name)

    requests.patch(f"http://backend-service/projects/{project_id}/state", json={"state": "[4/4] Done"})
    sio.emit('projectState', {'projectId': project_id, 'state': '[4/4] Done'})

    print("Aggregator job finished. Done!")

    requests.patch(f"http://backend-service/projects/{project_id}/logs", json={"logs": "Aggregator job finished. Done!"})
    sio.emit('projectLogs', {'projectId': project_id, 'logs': 'Aggregator job finished. Done!'})

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Error: ", sys.exc_info()[0])
        sio.disconnect()
        exit(5)
    sio.disconnect()
    exit()
        