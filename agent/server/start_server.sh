cd /root/edge-cluster-scheduler/agent/server || exit 1
python3 /root/edge-cluster-scheduler/agent/server/flask_openai_server.py --rkllm_model_path=/root/models/Qwen3-1.7B-rk3588-w8a8.rkllm --target_platform=rk3588
