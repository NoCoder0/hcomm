/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "orion_adpt_utils.h"

#include <cstring>

// Orion
#include "adapter_rts_common.h"
#include "orion_adapter_hccp.h"
#include "tp_manager.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "urma_api.h"

// 从 dl_urma_function.c 导出的 URMA 封装，避免 include dl_urma_function.h 引发 list_entry 等宏冲突
extern "C" {
    int RsUbApiInit(void);
    urma_device_t **RsUrmaGetDeviceList(int *numDevices);
    urma_device_t *RsUrmaGetDeviceByEid(urma_eid_t eid, urma_transport_type_t type);
    void RsUrmaFreeDeviceList(urma_device_t **deviceList);
    urma_eid_info_t *RsUrmaGetEidList(urma_device_t *dev, uint32_t *cnt);
    void RsUrmaFreeEidList(urma_eid_info_t *eidList);
}

namespace hcomm {

static bool TryResolveEidToPrimary(const Hccl::Eid &inputEid, Hccl::Eid &outputEid)
{
    (void)RsUbApiInit();

    int devNum = 0;
    urma_device_t **devList = RsUrmaGetDeviceList(&devNum);
    if (devList == NULL || devNum == 0) {
        return false;
    }

    urma_eid_t urmaEid;
    (void)memcpy(urmaEid.raw, inputEid.raw, Hccl::URMA_EID_LEN);

    urma_device_t *dev = RsUrmaGetDeviceByEid(urmaEid, URMA_TRANSPORT_UB);
    if (dev == NULL) {
        RsUrmaFreeDeviceList(devList);
        return false;
    }

    uint32_t eidCount = 0;
    urma_eid_info_t *eidList = RsUrmaGetEidList(dev, &eidCount);
    if (eidList == NULL || eidCount == 0) {
        RsUrmaFreeDeviceList(devList);
        return false;
    }

    // 在 EID 列表中寻找与 bonding EID 不同的首个 EID（即为物理端口 primary EID）
    for (uint32_t i = 0; i < eidCount; i++) {
        if (memcmp(eidList[i].eid.raw, inputEid.raw, Hccl::URMA_EID_LEN) != 0) {
            (void)memcpy(outputEid.raw, eidList[i].eid.raw, Hccl::URMA_EID_LEN);
            RsUrmaFreeEidList(eidList);
            RsUrmaFreeDeviceList(devList);
            return true;
        }
    }

    // 若全部相同（非 bonding 设备），直接用自身
    (void)memcpy(outputEid.raw, eidList[0].eid.raw, Hccl::URMA_EID_LEN);
    RsUrmaFreeEidList(eidList);
    RsUrmaFreeDeviceList(devList);
    return true;
}

const char *CommAddrTypeToStr(CommAddrType type)
{
    switch (type) {
        case COMM_ADDR_TYPE_IP_V4:
            return "COMM_ADDR_TYPE_IP_V4";
        case COMM_ADDR_TYPE_IP_V6:
            return "COMM_ADDR_TYPE_IP_V6";
        case COMM_ADDR_TYPE_EID:
            return "COMM_ADDR_TYPE_EID";
        case COMM_ADDR_TYPE_ID:
            return "COMM_ADDR_TYPE_ID";
        case COMM_ADDR_TYPE_RESERVED:
            return "COMM_ADDR_TYPE_RESERVED";
        default:
            return "UNKNOWN_COMM_ADDR_TYPE";
    }
}

HcclResult CommAddrToIpAddress(const CommAddr &commAddr, Hccl::IpAddress &ipAddr)
{
    if (commAddr.type != COMM_ADDR_TYPE_IP_V4 && commAddr.type != COMM_ADDR_TYPE_IP_V6 && commAddr.type != COMM_ADDR_TYPE_EID) {
        if (commAddr.type == COMM_ADDR_TYPE_ID || commAddr.type == COMM_ADDR_TYPE_RESERVED) {
            HCCL_ERROR("[%s] failed, comm address type[%d][%s] is not supported.", __func__, commAddr.type,
                CommAddrTypeToStr(commAddr.type));
        } else {
            HCCL_ERROR("[%s] failed, comm address type[%d][%s] is invalid.", __func__, commAddr.type,
                CommAddrTypeToStr(commAddr.type));
        }
        return HCCL_E_NOT_SUPPORT;
    }

    Hccl::BinaryAddr binAddr;
    int32_t family = AF_INET6;
    if (commAddr.type == COMM_ADDR_TYPE_IP_V4) {
        binAddr.addr = commAddr.addr;
        int32_t family = AF_INET;
        ipAddr = Hccl::IpAddress(binAddr, family);
        return HCCL_SUCCESS;
    }

    if (commAddr.type == COMM_ADDR_TYPE_EID){
        Hccl::Eid inputEid;
        s32 sret = memcpy_s(inputEid.raw, Hccl::URMA_EID_LEN, commAddr.eid, Hccl::URMA_EID_LEN);
        CHK_PRT_RET(sret != EOK, HCCL_ERROR("memcpy failed. errorno[%d]:", sret), HCCL_E_MEMORY);

        Hccl::Eid resolvedEid;
        if (TryResolveEidToPrimary(inputEid, resolvedEid)) {
            ipAddr = Hccl::IpAddress(resolvedEid);
        } else {
            ipAddr = Hccl::IpAddress(inputEid);
        }
        return HCCL_SUCCESS;
    }

    binAddr.addr6 = commAddr.addr6;
    ipAddr = Hccl::IpAddress(binAddr, family);
    return HCCL_SUCCESS;
}

HcclResult IpAddressToCommAddr(const Hccl::IpAddress &ipAddr, CommAddr &commAddr)
{
    int32_t family = ipAddr.GetFamily();
    const auto &binAddr = ipAddr.GetBinaryAddress();

    if (family == AF_INET) {
        commAddr.addr = binAddr.addr;
        commAddr.type = COMM_ADDR_TYPE_IP_V4;
        return HcclResult::HCCL_SUCCESS;
    }

    commAddr.addr6 = binAddr.addr6;
    commAddr.type = COMM_ADDR_TYPE_IP_V6;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommProtocolToLinkProtocol(CommProtocol commProtocol, Hccl::LinkProtocol &linkProtocol)
{
    switch (commProtocol) {
        case COMM_PROTOCOL_UBC_CTP:
            linkProtocol = Hccl::LinkProtocol::UB_CTP;
            break;
        case COMM_PROTOCOL_UBC_TP:
            linkProtocol = Hccl::LinkProtocol::UB_TP;
            break;
        case COMM_PROTOCOL_ROCE:
            linkProtocol = Hccl::LinkProtocol::ROCE;
            break;
        case COMM_PROTOCOL_HCCS:
            linkProtocol = Hccl::LinkProtocol::HCCS;
            break;
        case COMM_PROTOCOL_UB_MEM:
            linkProtocol = Hccl::LinkProtocol::UB_MEM;
            break;
        case COMM_PROTOCOL_PCIE:
            linkProtocol = Hccl::LinkProtocol::PCIE;
            break;
        case COMM_PROTOCOL_UBOE:
            linkProtocol = Hccl::LinkProtocol::UBOE;
            break;
        case COMM_PROTOCOL_UBG:
            linkProtocol = Hccl::LinkProtocol::UBG;
            break;
        default:
            HCCL_ERROR("[%s] Invalid CommProtocol[%u]", __func__, commProtocol);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CommAddrTypeToHcclAddressType(CommAddrType commAddrType, HcclAddressType &hcclAddressType)
{
    switch (commAddrType) {
        case COMM_ADDR_TYPE_IP_V4:
            hcclAddressType = HCCL_ADDR_TYPE_IP_V4;
            break;
        case COMM_ADDR_TYPE_IP_V6:
            hcclAddressType = HCCL_ADDR_TYPE_IP_V6;
            break;
        default:
            HCCL_ERROR("[%s] Invaild CommAddrType[%u]", __func__, commAddrType);
            return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

Hccl::LinkData BuildDefaultLinkData()
{
    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::HOST_NET;
    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::ROCE;
    Hccl::IpAddress locAddr;
    Hccl::IpAddress rmtAddr;
    uint32_t locDevPhyId = 0;
    uint32_t rmtDevPhyId = 0;
    return Hccl::LinkData(
        portDeploymentType,
        linkProtocol, 
        locDevPhyId, rmtDevPhyId,
        locAddr, rmtAddr
    );
}

static HcclResult EndpointLocTypeToPortDeploymentType(const EndpointLocType locType,
    Hccl::PortDeploymentType &deployType)
{
    switch (locType) {
        case EndpointLocType::ENDPOINT_LOC_TYPE_HOST:
            deployType = Hccl::PortDeploymentType::HOST_NET;
            break;
        case EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE:
            deployType = Hccl::PortDeploymentType::DEV_NET;
            break;
        default:
            HCCL_ERROR("[%s] unknown type of EndpointLocType[%d]", __func__, locType);
            return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult EndpointDescPairToLinkData(const EndpointDesc &locEp, const EndpointDesc &rmtEp,
    Hccl::LinkData &linkData, u32 reuseIdx)
{
    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::INVALID;
    CHK_RET(EndpointLocTypeToPortDeploymentType(locEp.loc.locType, portDeploymentType));

    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::INVALID;
    CHK_RET(CommProtocolToLinkProtocol(locEp.protocol, linkProtocol));

    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(locEp.commAddr, locAddr));
    CHK_RET(CommAddrToIpAddress(rmtEp.commAddr, rmtAddr));

    uint32_t locDevPhyId = locEp.loc.device.devPhyId;
    uint32_t rmtDevPhyId = rmtEp.loc.device.devPhyId;

    // 开源开放架构下comms层级不感知通信域层级的rank信息
    // 当前复用orion数据结构故使用devId替换
    linkData = Hccl::LinkData(
        portDeploymentType,
        linkProtocol, 
        locDevPhyId, rmtDevPhyId,
        locAddr, rmtAddr, reuseIdx
    );

    return HCCL_SUCCESS;
}

HcclResult EndpointDescPairToLinkDataWithRankIds(const uint32_t myRank, const uint32_t rmtRank,
    const EndpointDesc &locEp, const EndpointDesc &rmtEp, Hccl::LinkData &linkData, uint32_t devicePhyId, uint32_t remoteDevicePhyId,
    u32 reuseIdx)
{
    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::INVALID;
    CHK_RET(EndpointLocTypeToPortDeploymentType(locEp.loc.locType, portDeploymentType));

    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::INVALID;
    CHK_RET(CommProtocolToLinkProtocol(locEp.protocol, linkProtocol));

    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(locEp.commAddr, locAddr));
    CHK_RET(CommAddrToIpAddress(rmtEp.commAddr, rmtAddr));

    // 临时方案，为支持开源开放与orion通信域混跑，复用orion数据结构，添加rank信息
    linkData = Hccl::LinkData(
        portDeploymentType,
        linkProtocol, 
        myRank, rmtRank,
        locAddr, rmtAddr, devicePhyId, remoteDevicePhyId, reuseIdx
    );
    linkData.UpdateIpAddrWithPCIE();

    return HCCL_SUCCESS;
}

HcclResult PrepareUbConnBuildContext(const EndpointDesc &locEp, const EndpointDesc &rmtEp, uint32_t channelQos,
    UbConnBuildContext &ctx)
{
    CHK_RET(CommProtocolToLinkProtocol(locEp.protocol, ctx.protocol));
    CHK_RET(CommAddrToIpAddress(locEp.commAddr, ctx.locAddr));
    CHK_RET(CommAddrToIpAddress(rmtEp.commAddr, ctx.rmtAddr));
    CHK_RET(hrtGetDevice(&ctx.deviceLogicId));
    Hccl::TpManager::GetInstance(ctx.deviceLogicId).Init();
    if (channelQos > 7U) {
        HCCL_WARNING("[PrepareUbConnBuildContext] invalid channelQos[%u], expect [0, 7], use default qos[%u].",
            channelQos, Hccl::kRaUbGetTpInfoParamDefaultQos);
        ctx.qosPre = static_cast<u8>(Hccl::kRaUbGetTpInfoParamDefaultQos);
    } else {
        ctx.qosPre = static_cast<u8>(channelQos);
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm