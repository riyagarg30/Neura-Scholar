#!/bin/bash


function play_ping() {
    ansible all -i /inventory/myclusters/"${1}" -m ping
}


function play_pre_k8s() {
    ansible-playbook -i "/inventory/myclusters" /inventory/pre_k8s/pre_k8s_configure.yml
}

function play_build_k8s_cluster() {
    ansible-playbook -i "/inventory/myclusters/${1}" cluster.yml --become --become-user=root \
        > "/inventory/${1}.play.logs" 2>&1
}

function play_post_k8s() {
    ansible-playbook -i "/inventory/myclusters" /inventory/post_k8s/post_k8s_configure.yml
    ansible-playbook -i "/inventory/myclusters" /inventory/post_k8s/add_secrets.yml
    ansible-playbook -i "/inventory/myclusters" /inventory/post_k8s/add_drivers.yml
}

function play_mlflow_setup() {
    ansible-playbook -i "/inventory/myclusters" /inventory/mlflow/add_mlflow_platform.yml
}

function play_ray_cluster() {
    ansible-playbook -i "/inventory/myclusters" /inventory/ray-cluster/add_ray_operator.yml
}

function play_all() {
    play_ping;
    play_pre_k8s;
    play_build_k8s_cluster "$1";
    play_post_k8s;
    play_mlflow_setup;
    play_ray_cluster;
}

