#pragma once

#include <string>
#include <httplib.h>
#include "scheduler.h"
#include "spdlog/spdlog.h"

// 用户请求ai任务处理
const std::string QUSET_ROUTE = "/quest";
const std::string REGISTER_NODE_ROUTE = "/register_node";
const std::string HOTSTART_ROUTE = "/hot_start";

class HttpServer {
public:
    HttpServer(std::string ip, int port,string absoulte_config_path);
    bool Start();

private:
    static void HandleQuest(const httplib::Request &req, httplib::Response &res);
    static void HandleRegisterNode(const httplib::Request &req, httplib::Response &res);
    static void HandleHotStart(const httplib::Request &req, httplib::Response &res);

    std::string ip;
    int port;
};
