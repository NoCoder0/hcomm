/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "dlurma_function.h"
#include "hccl_next_dl.h"
#include "log.h"
 
namespace hcomm {
DlUrmaFunction &DlUrmaFunction::GetInstance()
{
    static DlUrmaFunction hcclDlUrmaFunction;
    return hcclDlUrmaFunction;
}
 
DlUrmaFunction::DlUrmaFunction() : handle_(nullptr)
{
}
 
DlUrmaFunction::~DlUrmaFunction()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ != nullptr) {
        (void)HcclNextDlclose(handle_);
        handle_ = nullptr;
    }
}
 
HcclResult DlUrmaFunction::DlUrmaFunctionApiInit()
{
    dlUrmaPostJettySendWr = (urma_status_t(*)(urma_jetty_t *jetty, urma_jfs_wr_t *wr, urma_jfs_wr_t **bad_wr))
        HcclNextDlsym(handle_, "urma_post_jetty_send_wr");
    CHK_SMART_PTR_NULL(dlUrmaPostJettySendWr);
 
    dlUrmaPollJfc = (int(*)(urma_jfc_t *jfc, int cr_cnt, urma_cr_t *cr))
        HcclNextDlsym(handle_, "urma_poll_jfc");
    CHK_SMART_PTR_NULL(dlUrmaPollJfc);
    return HCCL_SUCCESS;
}

HcclResult DlUrmaFunction::DlUrmaFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclNextDlopen("liburma.so.0", RTLD_NOW);
        if (handle_ == nullptr) {
            handle_ = HcclNextDlopen("/lib64/liburma.so.0", RTLD_NOW);
        }
        const char* errMsg = dlerror();
        CHK_PRT_RET(handle_ == nullptr, HCCL_ERROR("dlopen [%s] failed, %s", "liburma.so.0",\
            (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),\
            HCCL_E_OPEN_FILE_FAILURE);
    }

    // liburma 的 vendor provider 仅在 urma_init() 调用后才会挂载到 g_driver_list；
    // 不调用 urma_init() 时 urma_post_jetty_send_wr / urma_poll_jfc 都会在未初始化的
    // liburma 上工作。幂等调用：URMA_EEXIST(-17) 视为成功。
    static bool gDlUrmaInited = false;
    if (!gDlUrmaInited) {
        auto dlUrmaInit = (int (*)(urma_init_attr_t *))HcclNextDlsym(handle_, "urma_init");
        if (dlUrmaInit != nullptr) {
            int initRet = dlUrmaInit(NULL);
            if (initRet != 0 && initRet != -17) {  // -17 == URMA_EEXIST
                HCCL_WARNING("[DlUrmaFunctionInit] urma_init returned %d, data plane may fail.", initRet);
            }
        } else {
            HCCL_WARNING("[DlUrmaFunctionInit] urma_init symbol not found in liburma.so.");
        }
        gDlUrmaInited = true;
    }

    CHK_RET(DlUrmaFunctionApiInit());
    return HCCL_SUCCESS;
}
}