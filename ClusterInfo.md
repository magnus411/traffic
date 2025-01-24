10er Rommet pc Info:
PC01 - 592441549 - 10.10.20.75 - 04-0E-3C-3D-4D-98
pc02 - 1906823289 - 10.10.20.24 - 04-0E-3C-3D-05-CC
pc03 - 1431395807 - 10.10.20.49 - 6C-02-E0-3E-DE-80
pc04 - 1443729037 - 10.10.20.73 - 6C-02-E0-41-7E-A1
pc05 - 1512504939
pc06 - 1483750324
pc07 - 438560108
pc08 - 1724020647
pc09 - 1449998128
pc10 - 1686884851

(local data, dont mean shit)

Install requirements:
pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.26.4
pip install "ray[all]"
pip install wandb
pip install traci
pip install Pillow
pip install pyarrow

Starting cluster:
ray start --head --port=6379 --num-cpus=6 --num-gpus=1 --dashboard-host=10.10.20.75 (or head IP)

ray start --address="10.10.20.73:6379" --num-cpus=14 --num-gpus=1

Submitt job:
ray job submit --address="http://10.10.20.73:8265" --runtime-env-json="{\"working_dir\": \"C:\\\\Users\\\\support\\\\Desktop\\\\RLTraffic\"}" -- python main.py

Shared dir user:
rlworker
LJKSkjasij2oa0921"3jas0
(not important dir, everyone can have access)

Traffic lights in sim:
cluster*155251477_4254141700_6744555967_78275
cluster_1035391877_6451996796
cluster_163115359_2333391359_2333391361_78263*#1more
cluster_105371_4425738061_4425738065_4425738069
cluster_105372_4916088731_4916088738_4916088744
