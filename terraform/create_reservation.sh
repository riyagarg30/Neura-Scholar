#!/bin/bash

RESERVATION="MLOps_reservation_project22"


function create_reservation() {

    START_DATE=$(date -u +"%Y-%m-%d %H:%M") 
    END_DATE=$(date -u -v+6H +"%Y-%m-%d %H:%M") 

    output=$(
        openstack reservation lease create \
          --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_rtx_6000"]' \
          --start-date "$START_DATE" \
          --end-date "$END_DATE" \
          "$RESERVATION"
    )

    RESERVATION_ID=$(echo "$output" | grep '"id":' | awk -F'"' '{print $4}' | tail -1)

    while true; do
        output=$(openstack reservation lease show $RESERVATION_ID)
        printf "\r%s" "$(echo "$output" | grep -i '| status\s*|' | xargs)"
        echo "$output" | grep -i '| status\s*| ACTIVE' && break
        sleep 2
    done

    printf "RESERVATION_ID : %s\n" "$RESERVATION_ID"
}


function delete_reservation() {
    openstack reservation lease delete $RESERVATION
}
