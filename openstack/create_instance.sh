#!/bin/bash

PRIVATE_NETWORK="private_net_project22"
SUBNET="subnet_project22"

KEY_NAME="Preetham Rakshith"
FLOATING_IP="129.114.24.244" 
FLAVOR="m1.medium"


function create_instance() {

    NAME="$1"
    IP_ADDRESS="$2"

    PORT_NAME="${NAME}_port_project22"
    openstack port create \
        --network $PRIVATE_NETWORK \
        --fixed-ip subnet="$SUBNET",ip-address="$IP_ADDRESS" \
        --disable-port-security \
        "$PORT_NAME"

    openstack server create \
        --image "CC-Ubuntu24.04" \
        --flavor "$FLAVOR" \
        --network sharednet1 \
        --port "$PORT_NAME" \
        --security-group default \
        --security-group allow-ssh \
        --security-group allow-http-80 \
        --key-name "$KEY_NAME" \
        --user-data config-hosts.yaml \
        "${NAME}_project22"
}


function delete_instance() {
    NAME="$1"
    openstack server delete "${NAME}_project22"
    openstack port delete "${NAME}_port_project22"
}

#openstack server add floating ip "$NAME" "$FLOATING_IP"

