#!/bin/bash

RESERVATION="MLOps_reservation_project22"

function wait_for_reservation() {

    LEASE_ID="$1"

    while true; do
        output=$(openstack reservation lease show -f json -c reservations $LEASE_ID)
        printf "\r%s" "$(echo "$output" | jq -r '.reservations | fromjson | .status')"
    done
}

function create_reservation() {

    START_DATE=$(date -u +"%Y-%m-%d %H:%M") 
    END_DATE=$(date -u -v+6H +"%Y-%m-%d %H:%M") 

    OUTPUT=$(
    openstack reservation lease create \
      --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_rtx_6000"]' \
      --start-date "$START_DATE" \
      --end-date "$END_DATE" \
      --format shell \
      --column reservations \
      "$RESERVATION"
    )
    echo "$( echo $OUTPUT | grep -Po '\{[^}]*\}')"
    echo $OUTPUT
    LEASE_ID="$(echo "$OUTPUT" | jq -r '.lease_id')"
    echo $OUTPUT

    while true; do
        output=$(openstack reservation lease show $LEASE_ID)
        printf "\r%s" "$(echo "$output" | grep -i '| status\s*|' | xargs)"
        echo "$output" | grep -i '| status\s*| ACTIVE' && break
        sleep 2
    done

    RESERVATION_ID="$(echo "$OUTPUT" | jq -r '.id')"
    printf "RESERVATION_ID : %s\n" "$RESERVATION_ID"
}


function delete_reservation() {
    openstack reservation lease delete $RESERVATION
}
