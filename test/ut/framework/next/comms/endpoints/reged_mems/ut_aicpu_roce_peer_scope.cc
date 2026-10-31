/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <sstream>
#include <map>
#include <unordered_map>
#include "gtest/gtest.h"
#include "hccl_network.h"
#include "local_rdma_rma_buffer.h"
#define private public
#include "aicpu_ts_roce_mem.h"
#undef private
#include "exchange_rdma_buffer_dto.h"

namespace {
constexpr uint64_t ADDRESS = 0x2a0040000000;
constexpr uint64_t BYTES = 0x100000;

template <typename T>
void Append(std::vector<char> &bytes, const T &value)
{
    const char *begin = reinterpret_cast<const char *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

EndpointDesc Endpoint(bool host, uint32_t process)
{
    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_ROCE;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpoint.commAddr.addr.s_addr = 0x1d02a814;
    endpoint.loc.locType = host ? ENDPOINT_LOC_TYPE_HOST : ENDPOINT_LOC_TYPE_DEVICE;
    if (!host) {
        endpoint.loc.device.devPhyId = 12;
    }
    std::memcpy(endpoint.raws, &process, sizeof(process));
    return endpoint;
}

std::vector<char> Descriptor(const EndpointDesc &endpoint, uint32_t key)
{
    std::vector<char> bytes;
    if (endpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        Hccl::ExchangeRdmaBufferDto dto(ADDRESS, BYTES, key, "same-va-distinct-process");
        Hccl::BinaryStream stream;
        dto.Serialize(stream);
        stream.Dump(bytes);
    } else {
        Append(bytes, uint8_t{1});
        Append(bytes, ADDRESS);
        Append(bytes, BYTES);
        Append(bytes, ADDRESS);
        Append(bytes, uint32_t{1});
        Append(bytes, key);
    }
    Append(bytes, endpoint);
    return bytes;
}

void CheckPeerIsolation(bool host)
{
    hcomm::AicpuTsRoceRegedMemMgr manager(nullptr, nullptr);
    const auto first = Endpoint(host, 101);
    const auto second = Endpoint(host, 102);
    auto left = Descriptor(first, 17);
    auto right = Descriptor(second, 29);
    HcommMem memory{};
    ASSERT_EQ(manager.MemoryImport(left.data(), left.size(), &memory), HCCL_SUCCESS);
    ASSERT_EQ(manager.MemoryImport(right.data(), right.size(), &memory), HCCL_SUCCESS);
    std::vector<RoceMemDetails> details;
    manager.GatherRemoteMemDetails(details, &first);
    ASSERT_EQ(details.size(), 1U);
    EXPECT_EQ(details[0].addr, ADDRESS);
    EXPECT_EQ(details[0].key, 17U);
    details.clear();
    manager.GatherRemoteMemDetails(details, &second);
    ASSERT_EQ(details.size(), 1U);
    EXPECT_EQ(details[0].addr, ADDRESS);
    EXPECT_EQ(details[0].key, 29U);
    ASSERT_EQ(manager.MemoryUnimport(left.data(), left.size()), HCCL_SUCCESS);
    details.clear();
    manager.GatherRemoteMemDetails(details, &second);
    ASSERT_EQ(details.size(), 1U);
    EXPECT_EQ(details[0].key, 29U);
    details.clear();
    manager.GatherRemoteMemDetails(details, &first);
    EXPECT_TRUE(details.empty());
    EXPECT_EQ(manager.MemoryUnimport(right.data(), right.size()), HCCL_SUCCESS);
}

TEST(AicpuRocePeerScope, SameHostAddressRetainsIndependentKeysAndUnimport)
{
    CheckPeerIsolation(true);
}

TEST(AicpuRocePeerScope, SamePhysicalDeviceAddressRetainsIndependentKeysAndUnimport)
{
    CheckPeerIsolation(false);
}

TEST(AicpuRocePeerScope, UnknownPeerDoesNotExposeAnotherPeersMemory)
{
    hcomm::AicpuTsRoceRegedMemMgr manager(nullptr, nullptr);
    auto first = Endpoint(true, 101);
    auto unknown = Endpoint(true, 999);
    auto bytes = Descriptor(first, 17);
    HcommMem memory{};
    ASSERT_EQ(manager.MemoryImport(bytes.data(), bytes.size(), &memory), HCCL_SUCCESS);
    std::vector<RoceMemDetails> remote;
    manager.GatherRemoteMemDetails(remote, &unknown);
    EXPECT_TRUE(remote.empty());
    EXPECT_EQ(manager.MemoryUnimport(bytes.data(), bytes.size()), HCCL_SUCCESS);
}
} // namespace
