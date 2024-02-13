import os

# os.system("'/home/mark/Machine Learning/prometheus-2.45.0.linux-amd64/prometheus' --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml &")
# os.system("/usr/share/grafana/bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini --homepath /usr/share/grafana web &")

"""Tensor board tag regex: ^(?:(?!n_v|sampler_results).)*(loss|policy_reward_mean|vf|win_rate)(?:(?!n_v|sampler_results).)*$"""
os.system("tensorboard --logdir ./results &")
