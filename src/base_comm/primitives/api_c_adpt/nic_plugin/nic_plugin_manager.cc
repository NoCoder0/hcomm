/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nic_plugin_manager.h"

#include <acl/acl_rt.h>
#include <dirent.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "hcomm_result_defs.h"
#include "hcomm_adapter_runtime.h"
#include "log.h"
#include "param_check_pub.h"

namespace hcomm {
namespace {
constexpr const char *HCOMM_NIC_PLUGIN_DIR = "hcomm_plugin";
constexpr const char *HCOMM_NIC_PLUGIN_SO_ENV = "HCOMM_NIC_PLUGIN_SO";

std::once_flag &LoadOnce()
{
    static std::once_flag loadOnce;
    return loadOnce;
}

std::vector<std::unique_ptr<NicPluginEntry>> &LoadedPlugins()
{
    static std::vector<std::unique_ptr<NicPluginEntry>> loadedPlugins;
    return loadedPlugins;
}

std::unordered_map<CommProtocol, const NicPluginEntry *> &ProtocolPlugins()
{
    static std::unordered_map<CommProtocol, const NicPluginEntry *> protocolPlugins;
    return protocolPlugins;
}

bool EndsWithSo(const std::string &path)
{
    constexpr const char *suffix = ".so";
    constexpr size_t suffixLen = 3U;
    return path.size() >= suffixLen && path.compare(path.size() - suffixLen, suffixLen, suffix) == 0;
}

bool IsOpsHeaderValid(const CommAbiHeader &header, uint32_t magicWord, uint32_t version, const char *opsName)
{
    if (header.magicWord != magicWord) {
        HCCL_RUN_WARNING("[NicPlugin] %s magicWord[0x%08x] mismatch, expected[0x%08x].",
            opsName, header.magicWord, magicWord);
        return false;
    }
    if (header.version != version) {
        HCCL_RUN_WARNING("[NicPlugin] %s version[%u] mismatch, expected[%u].", opsName, header.version, version);
        return false;
    }
    if (header.size < sizeof(CommAbiHeader)) {
        HCCL_RUN_WARNING("[NicPlugin] %s size[%u] is smaller than ABI header[%zu].",
            opsName, header.size, sizeof(CommAbiHeader));
        return false;
    }
    return true;
}

bool ValidateEndpointOps(const HcommNicEndpointOps *ops)
{
    return ops != nullptr && IsOpsHeaderValid(ops->header, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD,
        HCOMM_NIC_ENDPOINT_OPS_VERSION, "endpoint ops");
}

bool ValidateChannelOps(const HcommNicChannelOps *ops)
{
    return ops != nullptr && IsOpsHeaderValid(ops->header, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD,
        HCOMM_NIC_CHANNEL_OPS_VERSION, "channel ops");
}

template <typename PluginOps>
bool IsPluginOpAvailable(const PluginOps *ops, size_t opOffset, size_t opSize)
{
    return ops != nullptr && ops->header.size >= opOffset + opSize;
}

void RegisterPluginProtocols(const NicPluginEntry *plugin)
{
    auto &protocolPlugins = ProtocolPlugins();
    for (uint32_t idx = 0; idx < plugin->info->protocolCount; ++idx) {
        const CommProtocol protocol = plugin->info->protocols[idx];
        auto iter = protocolPlugins.find(protocol);
        if (iter != protocolPlugins.end()) {
            HCCL_RUN_WARNING("[NicPlugin] protocol[%d] handler[%s] is overwritten by plugin[%s].",
                protocol,
                iter->second->info->name == nullptr ? "unknown" : iter->second->info->name,
                plugin->info->name == nullptr ? "unknown" : plugin->info->name);
        }
        protocolPlugins[protocol] = plugin;
        HCCL_RUN_INFO("[NicPlugin] protocol[%d] is handled by plugin[%s].",
            protocol, plugin->info->name == nullptr ? "unknown" : plugin->info->name);
    }
}

void *LoadSymbol(void *soHandle, const char *soPath, const char *symbol)
{
    dlerror();
    void *addr = dlsym(soHandle, symbol);
    const char *dlsymErr = dlerror();
    if (dlsymErr != nullptr || addr == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] dlsym %s from %s failed: %s.",
            symbol, soPath, dlsymErr == nullptr ? "unknown" : dlsymErr);
        return nullptr;
    }
    return addr;
}

void LoadOnePlugin(const std::string &path)
{
    if (path.empty()) {
        return;
    }
    void *soHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (soHandle == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] dlopen %s failed: %s.", path.c_str(), dlerror());
        return;
    }

    auto getInfo = reinterpret_cast<HcommNicPluginGetInfoFunc>(
        LoadSymbol(soHandle, path.c_str(), "HcommNicPluginGetInfo"));
    auto createEndpoint = reinterpret_cast<HcommNicPluginCreateEndpointFunc>(
        LoadSymbol(soHandle, path.c_str(), "HcommNicPluginCreateEndpoint"));
    auto createChannel = reinterpret_cast<HcommNicPluginCreateChannelFunc>(
        LoadSymbol(soHandle, path.c_str(), "HcommNicPluginCreateChannel"));
    if (getInfo == nullptr || createEndpoint == nullptr || createChannel == nullptr) {
        dlclose(soHandle);
        return;
    }

    const HcommNicPluginInfo *info = getInfo();
    if (!ValidatePluginInfo(path.c_str(), info, createEndpoint, createChannel)) {
        dlclose(soHandle);
        return;
    }

    std::unique_ptr<NicPluginEntry> plugin(new (std::nothrow) NicPluginEntry{soHandle, info, createEndpoint,
        createChannel});
    if (plugin == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] allocate plugin entry for %s failed.", path.c_str());
        dlclose(soHandle);
        return;
    }
    RegisterPluginProtocols(plugin.get());
    LoadedPlugins().emplace_back(std::move(plugin));
}

void LoadDefaultDirectory(const std::string &pluginDir)
{
    DIR *dir = opendir(pluginDir.c_str());
    if (dir == nullptr) {
        HCCL_RUN_INFO("[NicPlugin] plugin directory %s is unavailable.", pluginDir.c_str());
        return;
    }
    std::vector<std::string> soPaths;
    for (dirent *entry = readdir(dir); entry != nullptr; entry = readdir(dir)) {
        const std::string name(entry->d_name);
        if (name == "." || name == ".." || !EndsWithSo(name)) {
            continue;
        }
        soPaths.emplace_back(pluginDir + "/" + name);
    }
    closedir(dir);
    std::sort(soPaths.begin(), soPaths.end());
    for (const auto &path : soPaths) {
        LoadOnePlugin(path);
    }
}

void LoadExplicitPlugins(const char *envValue)
{
    if (envValue == nullptr || envValue[0] == '\0') {
        return;
    }
    const std::string paths(envValue);
    size_t start = 0;
    while (start <= paths.size()) {
        const size_t end = paths.find(':', start);
        const std::string path = paths.substr(start, end == std::string::npos ? std::string::npos : end - start);
        LoadOnePlugin(path);
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

void LoadPluginsOnce()
{
    uint32_t deviceCount = 0;
    const aclError ret = aclrtGetDeviceCount(&deviceCount);
    if (ret == ACL_SUCCESS && deviceCount != 0) {
        if (!IsHostNicPluginForceLoadEnabled()) {
            HCCL_RUN_INFO("[NicPlugin] plugin loading skipped, aclrtGetDeviceCount ret[%d], count[%u].",
                ret, deviceCount);
            return;
        }
        HCCL_RUN_WARNING("[NicPlugin] force loading Host NIC plugins with aclrtGetDeviceCount ret[%d], count[%u], "
            "env[%s]=1, test only.", ret, deviceCount, HCOMM_FORCE_HOST_NIC_PLUGIN_ENV);
    }

    const char *ascendHomePath = getenv("ASCEND_HOME_PATH");
    if (ascendHomePath != nullptr && ascendHomePath[0] != '\0') {
        LoadDefaultDirectory(std::string(ascendHomePath) + "/" + HCOMM_NIC_PLUGIN_DIR);
    } else {
        HCCL_RUN_INFO("[NicPlugin] ASCEND_HOME_PATH is empty, skip default plugin directory.");
        LoadExplicitPlugins(getenv(HCOMM_NIC_PLUGIN_SO_ENV));
    }
}

template <typename PluginOps>
void DestroyPluginCtx(PluginOps *ops, void *pluginCtx)
{
    if (ops != nullptr && IsPluginOpAvailable(ops, offsetof(PluginOps, destroy), sizeof(ops->destroy)) &&
        ops->destroy != nullptr) {
        ops->destroy(pluginCtx);
    }
}

template <typename PluginOps>
HcommResult InitPluginCtxOrDestroy(PluginOps *ops, void *pluginCtx)
{
    CHK_PTR_NULL(pluginCtx);
    CHK_PTR_NULL(ops);
    if (!IsPluginOpAvailable(ops, offsetof(PluginOps, init), sizeof(ops->init)) || ops->init == nullptr) {
        return HCCL_SUCCESS;
    }
    HcommResult ret = ops->init(pluginCtx);
    if (ret != HCCL_SUCCESS) {
        DestroyPluginCtx(ops, pluginCtx);
    }
    return ret;
}
} // namespace

bool ValidatePluginInfo(const char *soPath, const HcommNicPluginInfo *info,
    HcommNicPluginCreateEndpointFunc createEndpoint, HcommNicPluginCreateChannelFunc createChannel)
{
    if (info == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] %s exports null plugin info.", soPath);
        return false;
    }
    if (!IsOpsHeaderValid(info->header, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD,
        HCOMM_NIC_PLUGIN_INFO_VERSION, "plugin info")) {
        return false;
    }
    constexpr size_t requiredSize = offsetof(HcommNicPluginInfo, protocols) +
        sizeof(static_cast<HcommNicPluginInfo *>(nullptr)->protocols);
    if (info->header.size < requiredSize) {
        HCCL_RUN_WARNING("[NicPlugin] %s plugin info size[%u] is smaller than required[%zu].",
            soPath, info->header.size, requiredSize);
        return false;
    }
    if (info->protocolCount == 0 || info->protocolCount > HCOMM_NIC_PLUGIN_MAX_PROTOCOLS) {
        HCCL_RUN_WARNING("[NicPlugin] %s invalid protocolCount[%u].", soPath, info->protocolCount);
        return false;
    }
    if (createEndpoint == nullptr || createChannel == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] %s missing create endpoint/channel symbol.", soPath);
        return false;
    }
    for (uint32_t idx = 0; idx < info->protocolCount; ++idx) {
        const CommProtocol protocol = info->protocols[idx];
        if (protocol < COMM_PROTOCOL_HCCS || protocol > COMM_PROTOCOL_HCCS_ONLY) {
            HCCL_RUN_WARNING("[NicPlugin] %s invalid protocol[%d].", soPath, info->protocols[idx]);
            return false;
        }
    }
    return true;
}

void LoadAllNicPlugins()
{
    std::call_once(LoadOnce(), LoadPluginsOnce);
}

const NicPluginEntry *FindHostNicPlugin(CommProtocol protocol)
{
    LoadAllNicPlugins();
    const auto &protocolPlugins = ProtocolPlugins();
    auto iter = protocolPlugins.find(protocol);
    const NicPluginEntry *entry = iter == protocolPlugins.end() ? nullptr : iter->second;
    return entry;
}

HcommResult CreatePluginEndpoint(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    CHK_PTR_NULL(endpoint);
    CHK_PTR_NULL(endpointHandle);
    const NicPluginEntry *entry = FindHostNicPlugin(endpoint->protocol);
    if (entry == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    void *pluginCtx = nullptr;
    HcommNicEndpointOps *ops = nullptr;
    CHK_RET(static_cast<HcclResult>(entry->createEndpoint(endpoint, &pluginCtx, &ops)));
    CHK_PRT_RET(!ValidateEndpointOps(ops), HCCL_ERROR("[NicPlugin] invalid endpoint ops."), HCCL_E_PARA);
    HcommResult ret = InitPluginCtxOrDestroy(ops, pluginCtx);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    PluginEndpointCtx *ctx = new (std::nothrow) PluginEndpointCtx{ops, pluginCtx, entry};
    if (ctx == nullptr) {
        DestroyPluginCtx(ops, pluginCtx);
        return HCCL_E_MEMORY;
    }
    *endpointHandle = MAKE_PLUGIN_EP_HANDLE(ctx);
    return HCCL_SUCCESS;
}

HcommResult DestroyPluginEndpoint(EndpointHandle endpointHandle)
{
    PluginEndpointCtx *ctx = PLUGIN_EP_CTX(endpointHandle);
    CHK_PTR_NULL(ctx);
    DestroyPluginCtx(ctx->ops, ctx->ctx);
    delete ctx;
    return HCCL_SUCCESS;
}

HcommResult CreatePluginChannel(EndpointHandle endpointHandle, const HcommChannelDesc *channelDesc,
    ChannelHandle *channelHandle)
{
    PluginEndpointCtx *endpointCtx = PLUGIN_EP_CTX(endpointHandle);
    CHK_PTR_NULL(endpointCtx);
    CHK_PTR_NULL(endpointCtx->entry);
    CHK_PTR_NULL(channelDesc);
    CHK_PTR_NULL(channelHandle);
    void *pluginCtx = nullptr;
    HcommNicChannelOps *ops = nullptr;
    CHK_RET(static_cast<HcclResult>(
        endpointCtx->entry->createChannel(endpointCtx->ctx, channelDesc, &pluginCtx, &ops)));
    CHK_PRT_RET(!ValidateChannelOps(ops), HCCL_ERROR("[NicPlugin] invalid channel ops."), HCCL_E_PARA);
    HcommResult ret = InitPluginCtxOrDestroy(ops, pluginCtx);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    PluginChannelCtx *ctx = new (std::nothrow) PluginChannelCtx{ops, pluginCtx, endpointCtx->entry};
    if (ctx == nullptr) {
        DestroyPluginCtx(ops, pluginCtx);
        return HCCL_E_MEMORY;
    }
    *channelHandle = MAKE_PLUGIN_CH_HANDLE(ctx);
    return HCCL_SUCCESS;
}

HcommResult DestroyPluginChannel(ChannelHandle channelHandle)
{
    PluginChannelCtx *ctx = PLUGIN_CH_CTX(channelHandle);
    CHK_PTR_NULL(ctx);
    DestroyPluginCtx(ctx->ops, ctx->ctx);
    delete ctx;
    return HCCL_SUCCESS;
}

HcommResult UnsupportedPluginOp(const char *opName)
{
    HCCL_RUN_WARNING("[NicPlugin] plugin operation[%s] is not supported.", opName == nullptr ? "unknown" : opName);
    return HCCL_E_NOT_SUPPORT;
}

bool IsEndpointOpAvailable(const HcommNicEndpointOps *ops, size_t opOffset, size_t opSize)
{
    return ValidateEndpointOps(ops) && IsPluginOpAvailable(ops, opOffset, opSize);
}

bool IsChannelOpAvailable(const HcommNicChannelOps *ops, size_t opOffset, size_t opSize)
{
    return ValidateChannelOps(ops) && IsPluginOpAvailable(ops, opOffset, opSize);
}

} // namespace hcomm
