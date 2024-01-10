import os

# os.system("'/home/mark/Machine Learning/prometheus-2.45.0.linux-amd64/prometheus' --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml &")
# os.system("/usr/share/grafana/bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini --homepath /usr/share/grafana web &")
os.system("tensorboard --logdir '/home/mark/Machine Learning/Reinforcement Learning/Chess/results' &")
