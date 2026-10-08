# for slurm use
get_unused_port() {
    # Well-known ports end at 1023.  On Linux, dynamic ports start at 32768
    # (see /proc/sys/net/ipv4/ip_local_port_range).
    local MIN_PORT=10001
    local MAX_PORT=32767

    local USED_PORTS=$(netstat -a -n -t | tail -n +3 | tr -s ' ' | \
        cut -d ' ' -f 4 | sed 's/.*:\([0-9]\+\)$/\1/' | sort -n | uniq)

    # Generate random port numbers within the search range (inclusive) until we
    # find one that isn't in use.
    local RAN_PORT
    while
        RAN_PORT=$(shuf -i 10001-32767 -n 1)
        [[ "$USED_PORTS" =~ $RAN_PORT ]]
    do
        continue
    done

    echo $RAN_PORT
}

init_node_info() {
    export PRIMARY=$(hostname -s)
    SECONDARIES=$(scontrol show hostnames $SLURM_JOB_NODELIST | \
        grep -v $PRIMARY)

    ALL_NODES="$PRIMARY $SECONDARIES"
    export PRIMARY_PORT=$(get_unused_port)
    echo $PRIMARY $SECONDARIES $PRIMARY_PORT
}

init_node_info
