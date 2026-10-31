/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "endpoint_pair.h"
#include "roce_mem.h"

namespace {
constexpr uint64_t ADDRESS = 0x2a0040000000;
constexpr uint64_t DEVICE_ADDRESS = 0x12400000000;
constexpr uint64_t BYTES = 0x100000;

template <typename T>
void Append(std::vector<char> &bytes, const T &value)
{
    const char *begin = reinterpret_cast<const char *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

std::vector<char> Descriptor(bool deviceEndpoint, uint32_t memoryType)
{
    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_ROCE;
    endpoint.loc.locType = deviceEndpoint ? ENDPOINT_LOC_TYPE_DEVICE : ENDPOINT_LOC_TYPE_HOST;
    std::vector<char> bytes;
    if (deviceEndpoint) {
        Append(bytes, uint8_t{1});
        Append(bytes, ADDRESS);
        Append(bytes, BYTES);
        Append(bytes, DEVICE_ADDRESS);
        Append(bytes, memoryType);
        Append(bytes, uint32_t{17});
    } else {
        Hccl::ExchangeRdmaBufferDto dto(ADDRESS, BYTES, 17, "host-import-regression");
        Hccl::BinaryStream stream;
        dto.Serialize(stream);
        stream.Dump(bytes);
    }
    Append(bytes, endpoint);
    return bytes;
}

void CheckImport(bool deviceEndpoint, uint32_t memoryType, CommMemType expectedType)
{
    hcomm_experimental::RoceRegedMemMgr manager;
    auto bytes = Descriptor(deviceEndpoint, memoryType);
    HcommMem memory{};
    memory.type = COMM_MEM_TYPE_INVALID;
    ASSERT_EQ(manager.MemoryImport(bytes.data(), bytes.size(), &memory), HCCL_SUCCESS);
    EXPECT_EQ(memory.type, expectedType);
    EXPECT_EQ(reinterpret_cast<uint64_t>(memory.addr), deviceEndpoint ? DEVICE_ADDRESS : ADDRESS);
    EXPECT_EQ(memory.size, BYTES);
    EXPECT_EQ(manager.MemoryUnimport(bytes.data(), bytes.size()), HCCL_SUCCESS);
}

TEST(ExperimentalRoceMemoryImport, HostDescriptorSetsHostType)
{
    CheckImport(false, 0, COMM_MEM_TYPE_HOST);
}

TEST(ExperimentalRoceMemoryImport, DeviceDescriptorRetainsDeviceTypeAndRegisteredAddress)
{
    CheckImport(true, 0, COMM_MEM_TYPE_DEVICE);
}

TEST(ExperimentalRoceMemoryImport, DeviceHostDescriptorRetainsHostTypeAndRegisteredAddress)
{
    CheckImport(true, 1, COMM_MEM_TYPE_HOST);
}
} // namespace
