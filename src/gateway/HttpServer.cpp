#include "HttpServer.h"
#include "httplib.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <utility>
#include "DockerClient.h"
#include "TimeRecorder.h"

using json = nlohmann::json;

HttpServer::HttpServer(std::string ip, const int port, string absoulte_config_path) : ip(std::move(ip)), port(port) {
}

bool HttpServer::Start() {
    httplib::Server svr;
    // 注册路由
    svr.Post(QUSET_ROUTE, this->HandleQuest);
    svr.Post(REGISTER_NODE_ROUTE, this->HandleRegisterNode);
    svr.Post("/hot_start", this->HandleHotStart);
    spdlog::info("HttpServer started success，ip:{} port:{}",this->ip, this->port);
    auto result = svr.listen(this->ip, this->port);
    if (!result) {
        std::cerr << "HttpServer start failed！" << std::endl;
        return false;
    }
    return true;
}

// Define a function to modify or insert a key-value pair
void modifyOrInsert(httplib::Headers &headers,
                    const std::string &key, const std::string &newValue) {
    // Check if the key exists
    auto range = headers.equal_range(key);
    if (range.first != range.second) {
        // If found, modify the value for all matching keys
        for (auto it = range.first; it != range.second; ++it) {
            it->second = newValue;
        }
    } else {
        // If not found, insert a new key-value pair
        headers.emplace(key, newValue);
    }
}

//
// transfer request1
// example url: requet?taskid=0&real_url=hello/lxs
void HttpServer::HandleQuest(const httplib::Request &req, httplib::Response &res) {
    TimeRecord<std::chrono::milliseconds> time_record("HandleQuest");
    time_record.startRecord();

    TimeRecord<std::chrono::milliseconds> time_record_schedule("schedule");
    time_record_schedule.startRecord();

    // parse params
    auto task_type_str = req.get_param_value("taskid");
    auto real_url_param = req.get_param_value("real_url");
    TaskType task_type;
    // validate params
    try {
        task_type = StrToTaskType(task_type_str);
    } catch (const std::invalid_argument &e) {
        res.status = 400; // Bad Request
        res.set_content("invalid format taskid", "text/plain");
        return;
    }
    if (task_type == TaskType::Unknown) {
        res.status = 400; // Bad Request
        res.set_content("Missing taskid or real_url parameter", "text/plain");
        return;
    }
    if (real_url_param.empty()) {
        res.status = 400; // Bad Request
        res.set_content("Missing taskid or real_url parameter", "text/plain");
        return;
    }
    // get target_device_id
    optional<SrvInfo> srv_info_opt = Docker_scheduler::getOrCrtSrvByTType(task_type);
    if (srv_info_opt == nullopt) {
        res.status = 400; // Bad Request
        res.set_content("we can't get a useful srv", "text/plain");
        return;
    }
    // print in the end of quest
    time_record_schedule.endRecord();

    SrvInfo srv_info = srv_info_opt.value();
    string origin_host_ip = req.remote_addr;
    int origin_host_port = req.remote_port;
    string transfer_host_ip = req.local_addr;
    int transfer_host_port = req.local_port;
    string tgt_host_ip = srv_info.ip;
    int tgt_host_port = srv_info.port;

    httplib::Client cli(tgt_host_ip, tgt_host_port);
    // Forward the request to the target host.
    httplib::MultipartFormDataItems items;
    for (auto file: req.files) {
        items.push_back(file.second);
    }

    httplib::Result response;
    try {
        response = cli.Post("/" + real_url_param, items);
    } catch (const std::exception &e) {
        std::cerr << req.path << "  Error sending request: " << e.what() << std::endl;
        res.status = 500;
        res.set_content("Internal Server Error", "text/plain");
        return;
    }

    if (response != nullptr && response->status != -1) {
        // return to client
        res.status = response->status;
        res.set_header("Content-Type", response->get_header_value("Content-Type"));

        time_record.endRecord();
        // append gateway_time to response

        spdlog::info(
            "URL: {},task_type_str:{}, real_url_param:{} origin_host_ip: {}:{}, transfer_host_ip: {}:{}, tgt_host_ip: {}:{}, duration_time:{}, time_record_schedule:{}",
            req.path,
            task_type_str,
            real_url_param,
            origin_host_ip,
            origin_host_port,
            transfer_host_ip,
            transfer_host_port,
            tgt_host_ip,
            tgt_host_port,
            time_record.getDuration(),
            time_record_schedule.getDuration()
        );
        nlohmann::json jsonData = nlohmann::json::parse(response->body);
        jsonData["gateway_time"] = (double) (time_record.getDuration());
        res.body = jsonData.dump();
    } else {
        res.status = 502;
        time_record.endRecord();
        res.set_content("Bad Gateway", "text/plain");
        spdlog::error(
            "URL: {},task_type_str:{}, real_url_param:{} origin_host_ip: {}:{}, transfer_host_ip: {}:{}, tgt_host_ip: {}:{}, duration_time:{}, Error:{}",
            req.path,
            task_type_str,
            real_url_param,
            origin_host_ip,
            origin_host_port,
            transfer_host_ip,
            transfer_host_port,
            tgt_host_ip,
            tgt_host_port,
            time_record.getDuration(),
            "Connection failed or response = nullptr"
        );
    }
}

void HttpServer::HandleRegisterNode(const httplib::Request &req, httplib::Response &res) {
    nlohmann::json jsonData = nlohmann::json::parse(req.body);
    Device device;
    device.parseJson(jsonData);
    Docker_scheduler::RegisNode(device);
    res.status = 200;
    res.set_content("Node registered successfully", "text/plain");
    spdlog::info("Node registered successfully, param:{}", req.body);
}

void HttpServer::HandleHotStart(const httplib::Request &req, httplib::Response &res) {
    auto task_type_str = req.get_param_value("taskid");
    TaskType task_type = StrToTaskType(task_type_str);
    bool ret = Docker_scheduler::HotStartAllNodeByTType(task_type);
    if (ret) {
        res.set_content("HotStart successfully", "text/plain");
        spdlog::info("HotStart successfully, task_type:{}", to_string(nlohmann::json(task_type)));
        res.status = 200;
    }else {
        res.set_content("HotStart failed", "text/plain");
        spdlog::info("HotStart failed, task_type:{}", to_string(nlohmann::json(task_type)));
        res.status = 400;
    }
}

