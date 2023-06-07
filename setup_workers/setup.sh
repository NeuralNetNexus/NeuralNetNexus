#!/bin/bash

# Deve ser sempre três ou mais, 1 master, 1 worker computing, 1 worker platofrm
minikube delete --profile neuralnetnexus

minikube start --nodes 4 -p neuralnetnexus

kubectl apply -f ../kubernetes/roles

# Associar uma role a cada nó
kubectl label nodes neuralnetnexus-m02 kubernetes.io/role=computing
kubectl label nodes neuralnetnexus-m03 kubernetes.io/role=auxilliar
kubectl label nodes neuralnetnexus-m04 kubernetes.io/role=platform

# Criar os namespaces
kubectl create namespace frontend
kubectl create namespace backend
kubectl create namespace communication
kubectl create namespace split
kubectl create namespace train-suppervisor
kubectl create namespace train
kubectl create namespace aggregator

# Como pode haver vários nós de computing e auxilliar focadas em tarefas diferentes, 
# é necessário criar labels para cada um para garantir que os pods sejam criados no nó correto

# Criar as labels para o computing
kubectl label nodes neuralnetnexus-m02 train=yes
kubectl label nodes neuralnetnexus-m02 aggregator=yes

# Criar as labels para o auxilliar
kubectl label nodes neuralnetnexus-m03 split=yes
kubectl label nodes neuralnetnexus-m03 train_suppervisor=yes

# Platform deployment
kubectl apply -f ../kubernetes/configmaps/*
kubectl apply -f ../kubernetes/deployments
kubectl apply -f ../kubernetes/services
kubectl apply -f ../kubernetes/ingress

kubectl get nodes -o wide