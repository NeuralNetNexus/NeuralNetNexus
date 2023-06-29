#!/bin/bash

microk8s enable dns storage ingress

# Create Roles
microk8s kubectl apply -f kubernetes/roles

# Add Nodes
#microk8s add-node

# Add Nodes to Roles
microk8s kubectl label nodes neuralnetnexus-desktop kubernetes.io/role=computing
microk8s kubectl label nodes surface-laptop kubernetes.io/role=computing
microk8s kubectl label nodes ubuntu2004 kubernetes.io/role=helper
microk8s kubectl label nodes fedora kubernetes.io/role=platform

# Criar as labels para o computing
microk8s kubectl label nodes neuralnetnexus-desktop computing=yessir
microk8s kubectl label nodes surface-laptop computing=yessir
microk8s kubectl label nodes ubuntu2004 computing=yessir
microk8s kubectl label nodes ubuntu-linux-22-04-desktop computing=yessir

# Criar as labels para o auxilliar
microk8s kubectl label nodes ubuntu2004 helper=yessir
microk8s kubectl label nodes surface-laptop helper=yessir

# Create objects
microk8s kubectl apply -f kubernetes/configmaps/*
microk8s kubectl apply -f kubernetes/pvc
microk8s kubectl apply -f kubernetes/statefulset
microk8s kubectl apply -f kubernetes/deployments
microk8s kubectl apply -f kubernetes/services
microk8s kubectl apply -f kubernetes/ingress