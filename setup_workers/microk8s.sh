#!/bin/bash

# Stop and remove any existing MicroK8s instance
#microk8s stop
#microk8s reset

# Start MicroK8s with multiple nodes
microk8s enable dns storage ingress
microk8s start #--nodes 4  -p neuralnetnexus

#microk8s status --wait-ready



# Label the nodes
microk8s kubectl apply -f kubernetes/roles

#microk8s kubectl delete role neuralnetnexus-m02
#microk8s kubectl delete role neuralnetnexus-m03
#microk8s kubectl delete role neuralnetnexus-m04


# Create namespaces
microk8s kubectl create namespace neuralnetnexus
microk8s kubectl create namespace frontend
microk8s kubectl create namespace backend
microk8s kubectl create namespace communication
microk8s kubectl create namespace split
microk8s kubectl create namespace train-suppervisor
microk8s kubectl create namespace train
microk8s kubectl create namespace aggregator

#microk8s kubectl delete namespace frontend
#microk8s kubectl delete namespace backend
#microk8s kubectl delete namespace communication
#microk8s kubectl delete namespace split
#microk8s kubectl delete namespace train-suppervisor
#microk8s kubectl delete namespace train
#microk8s kubectl delete namespace aggregator



# Label the computing nodes
#microk8s kubectl label nodes neuralnetnexus-m02 train=yes
#microk8s kubectl label nodes neuralnetnexus-m02 aggregator=yes

# Label the auxilliar nodes
#microk8s kubectl label nodes neuralnetnexus-m03 split=yes
#microk8s kubectl label nodes neuralnetnexus-m03 train_suppervisor=yes

# Apply configuration files
microk8s kubectl apply -f ../kubernetes/configmaps/*
microk8s kubectl apply -f ../kubernetes/deployments
microk8s kubectl apply -f ../kubernetes/services
microk8s kubectl apply -f ../kubernetes/ingress

# Check the nodes
microk8s kubectl get nodes -o wide

microk8s add-node