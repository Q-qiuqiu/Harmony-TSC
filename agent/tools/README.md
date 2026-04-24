# Agent Tools

## `cluster_resources_mcp.py`

Minimal MCP server over `stdio` that exposes one tool:

- `get_cluster_resources`
- `run_vision_task_on_node`

The tool reads cluster state from the gateway HTTP API:

- default URL: `http://127.0.0.1:6666/cluster/resources`
- execution URL: `http://127.0.0.1:6666/quest_on_node`

The default execution settings for `run_vision_task_on_node` are:

- `real_url=predict`
- `file_field_name=image`

Optional environment variables:

- `GATEWAY_HOST`
- `GATEWAY_PORT`
- `GATEWAY_CLUSTER_RESOURCES_PATH`
- `GATEWAY_QUEST_ON_NODE_PATH`

Run manually:

```bash
python3 /root/edge-cluster-scheduler/agent/tools/cluster_resources_mcp.py
```

Example MCP client config command:

```json
{
  "command": "python3",
  "args": ["/root/edge-cluster-scheduler/agent/tools/cluster_resources_mcp.py"],
  "env": {
    "GATEWAY_HOST": "127.0.0.1",
    "GATEWAY_PORT": "6666"
  }
}
```
