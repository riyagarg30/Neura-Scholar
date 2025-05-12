#!/bin/bash

RESERVATION="MLOps_late_late_reservation_project22"


function create_reservation() {

    openstack reservation lease create \
      --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_rtx_6000"]' \
      --start-date "$(date -u +"%Y-%m-%d %H:%M")" \
      --end-date "$(date -u -v+6H +"%Y-%m-%d %H:%M")" \
      "$RESERVATION"

    while true; do
        output=$(openstack reservation lease show "$RESERVATION")
        printf "\r%s" "$(echo "$output" | grep -i '| status\s*|' | xargs)"
        echo "$output" | grep -i '| status\s*| ACTIVE' && break
        sleep 2
    done

    RESERVATION_ID=$(openstack reservation lease show "$RESERVATION" | grep '"id":' | awk -F'"' '{print $4}' | tail -1)
    printf "RESERVATION_ID : %s\n" "$RESERVATION_ID"
}


function delete_reservation() {
    openstack reservation lease delete $RESERVATION
}
