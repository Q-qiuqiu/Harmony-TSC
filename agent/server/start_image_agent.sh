cd /root/edge-cluster-scheduler/agent/server || exit 1
python3 /root/edge-cluster-scheduler/agent/server/image_agent.py --host=0.0.0.0 --port=8082 --llm_api_url=http://192.168.58.3:8081/v1/chat/completions
