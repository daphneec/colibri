#!/bin/bash

# Run the bash job
JOBID=$(sbatch --parsable "$(pwd)/launch.sh")

# Wait until it is spawned
echo "[runit.sh] Spawned job $JOBID; waiting until scheduled..."
while [[ -z "$(squeue | grep $JOBID | grep R)" ]]; do sleep 1; done
# Wait one more sec for good measure, to ensure the log exists
sleep 1

# Attach to the log
log_path="$(pwd)/logs_${JOBID}.out"
echo "[runit.sh] Attaching to '$log_path'; if detached, you can re-attach using 'tail -f \"$log_path\"'"
tail -f "$log_path"

