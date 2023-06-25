#!/bin/bash

microk8s enable dns storage ingress

# Create Roles
microk8s kubectl apply -f kubernetes/roles

# Add Nodes
#microk8s add-node

# Add Nodes to Roles
#microk8s kubectl label nodes neuralnetnexus-desktop kubernetes.io/role=computing
#microk8s kubectl label nodes raspberrypitwo kubernetes.io/role=helper
#microk8s kubectl label nodes fedora kubernetes.io/role=platform

kubectl label nodes raspberrypitwo helper=yes
kubectl label nodes neuralnetnexus-desktop computing=yes

# Create objects
microk8s kubectl apply -f kubernetes/configmaps/*
microk8s kubectl apply -f kubernetes/pvc
microk8s kubectl apply -f kubernetes/statefulset
microk8s kubectl apply -f kubernetes/deployments
microk8s kubectl apply -f kubernetes/services
microk8s kubectl apply -f kubernetes/ingress