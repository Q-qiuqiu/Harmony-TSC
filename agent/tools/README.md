# Agent Tools

## `mcp_server.py`

Minimal MCP server over `stdio` that exposes tools backed by separate modules:

- `get_cluster_resources`
- `get_task_catalog`
- `run_vision_task_on_node`

Implementation modules:

- `cluster_resource_tool.py`
- `task_catalog_tool.py`
- `vision_execute_tool.py`

The tool reads cluster state from the gateway HTTP API:

- default URL: `http://127.0.0.1:6666/cluster/resources`
- execution URL: `http://127.0.0.1:6666/quest_on_node`

`get_task_catalog` reads compact task metadata from:

- default file: `/root/edge-cluster-scheduler/config_files/static_info.json`

It returns a flat list of items shaped like:

- `device_type`
- `model_name`
- `task_type`
- `overhead`

It can also be filtered by `available_device_types` so only models matching currently available node platforms are returned.

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
python3 /root/edge-cluster-scheduler/agent/tools/mcp_server.py
```

Example MCP client config command:

```json
{
  "command": "python3",
  "args": ["/root/edge-cluster-scheduler/agent/tools/mcp_server.py"],
  "env": {
    "GATEWAY_HOST": "127.0.0.1",
    "GATEWAY_PORT": "6666"
  }
}
```
