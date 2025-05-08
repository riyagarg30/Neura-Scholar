#!/bin/bash

RESERVATION="MLOps_reservation_project22"


function create_reservation() {

    START_DATE=$(date -u +"%Y-%m-%d %H:%M") 
    END_DATE=$(date -u -v+6H +"%Y-%m-%d %H:%M") 

    output=$(
        openstack reservation lease create \
          --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_k80"]' \
          --start-date "$START_DATE" \
          --end-date "$END_DATE" \
          --format json \
          --column reservations \
          "$RESERVATION"
    )

    LEASE_ID="$(echo "$output" | jq -r '.reservations | fromjson | .lease_id')"

    while true; do
        output=$(openstack reservation lease show $LEASE_ID)
        printf "\r%s" "$(echo "$output" | grep -i '| status\s*|' | xargs)"
        echo "$output" | grep -i '| status\s*| ACTIVE' && break
        sleep 2
    done

    RESERVATION_ID="$(echo "$output" | jq -r '.reservations | fromjson | .id')"
    printf "RESERVATION_ID : %s\n" "$RESERVATION_ID"
}


function delete_reservation() {
    openstack reservation lease delete $RESERVATION
}
