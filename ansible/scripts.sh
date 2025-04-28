#!/bin/bash


ansible all -i /inventory/myclusters -m ping


function ansible_play() {
    ansible-playbook -i "/inventory/myclusters" /inventory/pre_k8s/pre_k8s_configure.yml && \
    ansible-playbook -i "/inventory/myclusters" /inventory/docker-login.yml && \
    ansible-playbook -i "/inventory/myclusters" cluster.yml --become --become-user=root  && \
    ansible-playbook -i "/inventory/myclusters" /inventory/post_k8s/post_k8s_configure.yml
}

ansible_play > /inventory/play.logs 2>&1
