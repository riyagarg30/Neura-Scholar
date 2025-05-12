ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -NL 192.168.1.11:8265:192.168.1.10:8265 -L 192.168.1.11:8090:192.168.1.10:8080 cc@192.5.87.62 &

