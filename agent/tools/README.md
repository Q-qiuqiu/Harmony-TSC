# Agent Tools

## `cluster_resources_mcp.py`

Minimal MCP server over `stdio` that exposes one tool:

- `get_cluster_resources`

The tool reads cluster state from the gateway HTTP API:

- default URL: `http://127.0.0.1:6666/cluster/resources`

Optional environment variables:

- `GATEWAY_HOST`
- `GATEWAY_PORT`
- `GATEWAY_CLUSTER_RESOURCES_PATH`

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
