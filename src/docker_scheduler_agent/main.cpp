#include "MachineInfoCollector.h"
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include "device_type.h"
#include "device.h"

const char *kGatewayIp = "192.168.58.3";
const int kGatewayPort = 6666;
const int kAgentPort = 8000;

using json = nlohmann::json;
using namespace httplib;

static std::string BuildResult(const std::string &status, const json &v) {
    json j;
    j["status"] = status;
    j["result"] = v;
    return j.dump();
}

static std::string BuildSuccess(const json &v) { return BuildResult("success", v); }

static std::string BuildFailed(const json &v) { return BuildResult("failed", v); }

static bool RegisterNode(MachineInfoCollector &collector) {
    try {
        Client client(kGatewayIp, kGatewayPort);

        json j = {
                {"type",       AGENT_DEVICE_TYPE},
                {"global_id",  collector.GetGlobalId()},
                {"ip_address", collector.GetIp()},
                {"agent_port", kAgentPort},
        };

        Result result = client.Post("/register_node", j.dump(), "application/json");
        if (!result || result->status != OK_200) {
            std::cerr << "Failed to register node: " << result.error() << std::endl;
            return false;
        }

        std::cout << "Node registered successfully" << std::endl;
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Failed to register node: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    MachineInfoCollector collector(kGatewayIp, kGatewayPort);
    httplib::Server server;

    if (!RegisterNode(collector)) {
        return 1;
    }

    server.set_exception_handler([](const auto &req, auto &res, std::exception_ptr ep) {
        res.status = httplib::OK_200;
        std::string msg;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception &e) {
            msg = e.what();
        } catch (...) {
            msg = "unknown exception";
        }
        std::cerr << "exception: " << msg << std::endl;
        res.set_content(BuildFailed(msg), "application/json");
    });

    server.Get("/usage/device_info", [&collector](const httplib::Request &, httplib::Response &res) {
        DeviceStatus dev_info;
        dev_info.cpu_used = collector.GetCpuUsage();
        dev_info.mem_used = collector.GetMemoryUsage();
        dev_info.xpu_used = collector.GetNpuUsage();
        dev_info.net_latency = collector.GetNetLatency(); // ms
        dev_info.net_bandwidth = 1000; // Mbps (constant)
        std::string result = BuildSuccess(dev_info.to_json());
        res.set_content(result, "application/json");
    });

    std::cout << "Starting docker scheduler agent" << std::endl;
    if (!server.listen("0.0.0.0", kAgentPort)) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }

    return 0;
}
