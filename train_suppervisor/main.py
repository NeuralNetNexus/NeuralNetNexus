from os import getenv
from time import sleep

import yaml

from kubernetes import client, config

project_id = getenv('project_id')
pvc_name = getenv('pvc_name')

def create_job_object(job_name, image_name, pvc_name=None, mount_path=None, env_vars=None, completions=None, parallelism=None):
    # Construct the environment variables for the container
    env = []
    if env_vars is not None:
        for name, value in env_vars.items():
            env.append(client.V1EnvVar(name=name, value=str(value)))

    # Prepare the VolumeMount and Volume for the container and pod respectively if a PVC is provided
    volume_mounts = []
    volumes = []
    if pvc_name and mount_path:
        volume_mounts.append(client.V1VolumeMount(mount_path=mount_path, name='volume'))
        volumes.append(client.V1Volume(name='volume', persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)))

    # Define the job's container
    container = client.V1Container(
        name=job_name,
        image=image_name,
        volume_mounts=volume_mounts if volume_mounts else None,
        env=env if env else None,
    )

    # Define the job's template
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": job_name}),
        spec=client.V1PodSpec(restart_policy="Never", containers=[container], volumes=volumes if volumes else None),
    )

    # Define the job's spec
    if completions and parallelism:
        spec = client.V1JobSpec(
            completions=completions,
            parallelism=parallelism,
            template=template,
            backoff_limit=4,
        )
    else:
        spec = client.V1JobSpec(
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
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace="default")
    print("Job created. status='%s'" % str(api_response.status))

def get_job_status(api_instance, job_name):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=job_name,
            namespace="default")
        if api_response.status.succeeded is not None or \
                api_response.status.failed is not None:
            job_completed = True
        sleep(1)
        print("Job status='%s'" % str(api_response.status))

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    
    # {1} -> é o ID do projeto

    # Split Job
    split_job_name = f"split-job-{project_id}"
    env_vars = {"PARTS": 5}
    split_job = create_job_object(split_job_name, "rafaelxokito/neuralnetnexussplit:latest", pvc_name, "/app/database", env_vars)
    create_job(batch_v1, split_job)
    get_job_status(batch_v1, split_job_name)

    # TODO -> Ir buscar o número de batches do 'split' job e definir o número de 
    # completions e parallelism para o número de batches

    # Training Jobs
    train_job_name = f"train-job-{project_id}"
    train_job = create_job_object(train_job_name, "rafaelxokito/neuralnetnexustrain:latest", completions=env_vars["PARTS"], parallelism=env_vars["PARTS"])
    create_job(batch_v1, train_job)
    get_job_status(batch_v1, train_job_name)

    # Aggregator Job
    aggregator_job_name = f"aggregator-job-{project_id}"
    aggregator_job = create_job_object(aggregator_job_name, "rafaelxokito/neuralnetnexusagregator:latest")
    create_job(batch_v1, aggregator_job)
    get_job_status(batch_v1, aggregator_job_name)

if __name__ == '__main__':
    main()