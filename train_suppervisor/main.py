from os import getenv
from time import sleep
import requests

import yaml

from kubernetes import client, config

project_id = getenv('PROJECT_ID')
model = getenv('MODEL')

def create_volume_mounts(job_type):
    volume_mounts = []
    volumes = []

    volume_mounts.append(client.V1VolumeMount(mount_path="/app/datasets", name='datasets-data'))
    volumes.append(client.V1Volume(name='datasets-data', persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name="pvc-datasets")))

    if job_type == "split":
        return volume_mounts, volumes

    volume_mounts.append(client.V1VolumeMount(mount_path="/app/models", name='models-data'))
    volumes.append(client.V1Volume(name='models-data', persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name="pvc-models")))

    return volume_mounts, volumes

def create_job_object(job_name, image_name, env_vars=None, completions=None, parallelism=None):
    # Construct the environment variables for the container
    env = []
    if env_vars is not None:
        for name, value in env_vars.items():
            env.append(client.V1EnvVar(name=name, value=str(value)))

    # Define PVC(s) to mount
    job_type = job_name.split("-")[0]
    volume_mounts, volumes = create_volume_mounts(job_type)
    
    # Define the job's container
    container = client.V1Container(
        name=job_name,
        image=image_name,
        volume_mounts=volume_mounts if volume_mounts else None,
        env=env if env else None,
    )

    # Define the job's spec
    if completions and parallelism:
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
            metadata=client.V1ObjectMeta(labels={"app": job_name}),
            spec=client.V1PodSpec(restart_policy="Never", containers=[container], affinity=affinity, volumes=volumes if volumes else None),
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
            metadata=client.V1ObjectMeta(labels={"app": job_name}),
            spec=client.V1PodSpec(restart_policy="Never", containers=[container], affinity=affinity, volumes=volumes if volumes else None),
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
        print("Job status='%s'" % str(api_response.status))

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_incluster_config()
    batch_v1 = client.BatchV1Api()
    
    # {1} -> é o ID do projeto

    # =================  Split Job  ================= #

    split_job_name = f"split-job-{project_id}"
    image_name = "rafaelxokito/neuralnetnexussplit:latest"
    env_vars = {"PROJECT_ID": project_id}

    split_job = create_job_object(split_job_name, image_name, env_vars)
    create_job(batch_v1, split_job)
    get_job_status(batch_v1, split_job_name)

    # =================  Training Jobs  ================= #

    train_job_name = f"train-job-{project_id}"
    image_name = "rafaelxokito/neuralnetnexustrain:latest"
    n_splits = requests.api.get(f"http://backend-service/projects/{project_id}").json()["project"]["n_splits"]

    env_vars = {"PROJECT_ID": project_id, "MODEL": model}
    train_job = create_job_object(train_job_name, image_name, env_vars, completions=n_splits, parallelism=n_splits)
    create_job(batch_v1, train_job)
    get_job_status(batch_v1, train_job_name)

    # =================  Aggregator Job  ================= #

    aggregator_job_name = f"aggregator-job-{project_id}"
    image_name = "rafaelxokito/neuralnetnexusaggregator:latest"
    aggregator_job = create_job_object(aggregator_job_name, image_name, env_vars)
    create_job(batch_v1, aggregator_job)
    get_job_status(batch_v1, aggregator_job_name)

if __name__ == '__main__':
    main()