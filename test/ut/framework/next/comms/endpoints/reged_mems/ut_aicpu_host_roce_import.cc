/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <securec.h>
#include "exchange_rdma_buffer_dto.h"
#define private public
#include "aicpu_ts_roce_mem.h"
#include "aicpu_ts_roce_channel.h"
#undef private
using namespace hcomm;

namespace {
std::vector<char> HostRoceImportDescriptor()
{
    Hccl::ExchangeRdmaBufferDto dto(0x100000, 4096, 0x1234, "host-dram");
    Hccl::BinaryStream stream;
    dto.Serialize(stream);
    std::vector<char> result;
    stream.Dump(result);
    EndpointDesc ep{};
    EndpointDescInit(&ep, 1);
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    const auto *tail = reinterpret_cast<const char *>(&ep);
    result.insert(result.end(), tail, tail + sizeof(ep));
    return result;
}
}

TEST(AicpuHostRoceImport, PreservesHostAddressSizeTypeAndKey)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    auto descriptor = HostRoceImportDescriptor();
    HcommMem mem{};
    ASSERT_EQ(mgr.MemoryImport(descriptor.data(), descriptor.size(), &mem), HCCL_SUCCESS);
    EXPECT_EQ(mem.addr, reinterpret_cast<void *>(0x100000));
    EXPECT_EQ(mem.size, 4096U);
    EXPECT_EQ(mem.type, COMM_MEM_TYPE_HOST);
    std::vector<RoceMemDetails> remote;
    mgr.GatherRemoteMemDetails(remote);
    ASSERT_EQ(remote.size(), 1U);
    EXPECT_EQ(remote[0].key, 0x1234U);
    EXPECT_EQ(remote[0].devAddr, 0x100000U);
    EXPECT_EQ(mgr.MemoryUnimport(descriptor.data(), descriptor.size()), HCCL_SUCCESS);
}

TEST(AicpuHostRoceImport, RejectsTruncatedAndOversizedTagBeforeAllocation)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    auto descriptor = HostRoceImportDescriptor();
    HcommMem mem{};
    const size_t invalidSize = SIZE_MAX;
    ASSERT_EQ(memcpy_s(descriptor.data() + sizeof(u64) * 2 + sizeof(u32), sizeof(size_t),
        &invalidSize, sizeof(invalidSize)), EOK);
    EXPECT_EQ(mgr.MemoryImport(descriptor.data(), descriptor.size(), &mem), HCCL_E_PARA);
    descriptor.erase(descriptor.begin(), descriptor.begin() + 25);
    EXPECT_EQ(mgr.MemoryImport(descriptor.data(), descriptor.size(), &mem), HCCL_E_PARA);
}

TEST(AicpuHostRoceTag, MatchesHostSocketConfigForBothRolesAndPreservesDeviceTag)
{
    HcommChannelDesc desc{};
    desc.channelName = "read-demo";
    desc.port = 19517;
    AicpuTsRoceChannel channel(nullptr, desc);
    channel.localEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    channel.remoteEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    channel.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    ASSERT_EQ(inet_pton(AF_INET, "20.168.0.3", &channel.localEp_.commAddr.addr), 1);
    ASSERT_EQ(inet_pton(AF_INET, "20.168.0.19", &channel.remoteEp_.commAddr.addr), 1);
    std::string tag;
    channel.isLocalIpClient_ = true;
    ASSERT_EQ(channel.BuildSocketTagName(tag), HCCL_SUCCESS);
    EXPECT_EQ(tag, "read-demo_20.168.0.19_20.168.0.3_19517");
    channel.isLocalIpClient_ = false;
    ASSERT_EQ(channel.BuildSocketTagName(tag), HCCL_SUCCESS);
    EXPECT_EQ(tag, "read-demo_20.168.0.3_20.168.0.19_19517");
    channel.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ASSERT_EQ(channel.BuildSocketTagName(tag), HCCL_SUCCESS);
    EXPECT_EQ(tag, "read-demo");
}

