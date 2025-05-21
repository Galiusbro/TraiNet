import subprocess

total_shards = 2
processes = []

for shard_id in range(total_shards):
    proc = subprocess.Popen([
        "python", "worker.py", str(shard_id), str(total_shards)
    ])
    processes.append(proc)

for proc in processes:
    proc.wait() 