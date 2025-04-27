#!/bin/bash


# https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html

# USE HELM TO INSTALL NVIDIA GPU-OPERATOR FOR KUBERNETES.

curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh

kubectl create ns gpu-operator
kubectl label --overwrite ns gpu-operator pod-security.kubernetes.io/enforce=privileged

kubectl get nodes -o json | jq '.items[].metadata.labels | keys | any(startswith("feature.node.kubernetes.io"))'

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update

helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v25.3.0

sleep 60s && kubectl logs nvidia-cuda-validator-7l22d -n gpu-operator

# APPLY GPU TIME-SLICING PATCH.

kubectl create -n gpu-operator -f time-slicing-config-all.yaml

kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all", "default": "any"}}}}'

sleep 60s

# VERIFY TIME SLICING OF GPUs.

kubectl apply -f time-slicing-verification.yaml
kubectl get pods
kubectl logs deploy/time-slicing-verification
kubectl delete -f time-slicing-verification.yaml


