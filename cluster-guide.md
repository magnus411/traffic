ray up cluster.yaml

ray status

If you prefer not to use Ray's cluster launcher or are working within a local network without SSH, you can manually start Ray on each machine.

Start the Head Node

On the head node, start Ray in head mode:

ray start --head --port=6379 --num-cpus=8 --num-gpus=1 --dashboard-host=0.0.0.0
--dashboard-host=0.0.0.0: Makes the Ray dashboard accessible from other machines on the network.

Start Worker Nodes

On each worker node, connect to the head node:

ray start --address='HEAD_NODE_IP:6379' --redis-password='YOUR_REDIS_PASSWORD' --num-cpus=8 --num-gpus=1
ray start --address='HEAD_NODE_IP:6379' --redis-password='YOUR_REDIS_PASSWORD' --num-cpus=8 --num-gpus=1

Verify the Cluster
ray list nodes

As mentioned earlier, ensure that your_training_script.py and any necessary files are accessible to the head node. If you're running the script from the head node, worker nodes don't need direct access to the script unless you plan to run scripts directly on them.

http://HEAD_NODE_IP:8265
