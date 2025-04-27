
RESERVATION="rtx_project22"
FLOATING_IP="192.5.87.164"
NAME="gpu_node"

openstack reservation lease create \
  --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_rtx_6000"]' \
  "$RESERVATION"
  #--start-date "2025-04-25 18:00" \
  #--end-date "2025-04-25 18:00" \

while true; do
    output=$(openstack reservation lease show "$RESERVATION")
    echo "$output" | grep --color -i '| status\s*|'
    echo "$output" | grep -i '| status\s*| ACTIVE' && break
    sleep 2
done

RESERVATION_ID=$(openstack reservation lease show "$RESERVATION" | grep '"id":' | awk -F'"' '{print $4}' | tail -1)

openstack server create --image CC-Ubuntu24.04-CUDA --flavor baremetal --key-name "Preetham Rakshith" --network sharednet1 --security-group default --security-group allow-ssh --user-data config-hosts.yaml --hint reservation="$RESERVATION_ID" "${NAME}_project22"

openstack server show "${NAME}_project22"

while true; do
    output=$(openstack server show "${NAME}_project22")
    printf "\r%s" "$(echo "$output" | grep -i '| status\s*|' | xargs)"
    echo "$output" | grep -i '| status\s*| ACTIVE' && break
    sleep 2
done

openstack server add floating ip "${NAME}_project22" "$FLOATING_IP"

#openstack reservation delete test-reservation-pp2959
#openstack reservation lease delete test-reservation-pp2959

