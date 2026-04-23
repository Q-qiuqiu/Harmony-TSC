#ifndef DOCKER_SCHEDULER_AGENT_MACHINEINFOCOLLECTORBASE_H
#define DOCKER_SCHEDULER_AGENT_MACHINEINFOCOLLECTORBASE_H

#include <thread>
#include <mutex>
#include <string>
#include <queue>

const static size_t kCpuUsageQueueSize = 5;

struct CpuUsageInfo {
    uint64_t user;
    uint64_t system;
    uint64_t idle;
};

class MachineInfoCollectorBase {
public:
    MachineInfoCollectorBase(std::string gatewayIp, int gatewayPort)
            : gatewayIp(std::move(gatewayIp)), gatewayPort(gatewayPort) {
        StartCollect();
        for (size_t i = 0; i < kCpuUsageQueueSize; i++) {
            cpuUsageQueue.push_back(0.0);
        }
    }

    double GetCpuUsage();

    double GetMemoryUsage();

    double GetNetLatency();

    std::string GetIp();

    std::string GetGlobalId();

private:
    std::thread collectorThread;
    std::mutex collectorMutex;

    CpuUsageInfo prevCpuUsage{};
    CpuUsageInfo currCpuUsage{};
    std::deque<double> cpuUsageQueue;

    const std::string gatewayIp;
    const int gatewayPort{};
    double netLatency{};

    void StartCollect();

    void CollectThread();

    void CollectCpuUsage();

    void CollectNetLatency();
};

#endif // DOCKER_SCHEDULER_AGENT_MACHINEINFOCOLLECTORBASE_H
