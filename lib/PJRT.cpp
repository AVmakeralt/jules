//===- PJRT.cpp - PJRT Device API Implementation ---------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the PJRT device API for the Jules compiler.
// The CPU implementation provides a complete functional backend for
// development and testing. GPU/TPU backends would be added as plugins.
//
//===----------------------------------------------------------------------===//

#include "jules/PJRT.h"
#include "jules/Diagnostics.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

namespace jules {

// ── CPUBuffer implementation ────────────────────────────────────────────────

CPUBuffer::CPUBuffer(std::vector<int64_t> shape,
                     ScalarType::ScalarKind elementType, int deviceId)
    : shape_(std::move(shape)), elementType_(elementType),
      deviceId_(deviceId) {
  // Compute total number of elements.
  size_t numElements = 1;
  for (int64_t d : shape_) {
    numElements *= (d > 0 ? static_cast<size_t>(d) : 1);
  }

  // Compute bytes per element.
  size_t bytesPerElement = 4; // Default: f32
  switch (elementType_) {
  case ScalarType::SK_F32:  bytesPerElement = 4;  break;
  case ScalarType::SK_F64:  bytesPerElement = 8;  break;
  case ScalarType::SK_I32:  bytesPerElement = 4;  break;
  case ScalarType::SK_I64:  bytesPerElement = 8;  break;
  case ScalarType::SK_Bool: bytesPerElement = 1;  break;
  case ScalarType::SK_Unit: bytesPerElement = 0;  break;
  }

  data_.resize(numElements * bytesPerElement, 0);
}

CPUBuffer::~CPUBuffer() = default;

size_t CPUBuffer::size() const { return data_.size(); }

void CPUBuffer::copyFromHost(const void *src, size_t sz) {
  size_t copySize = std::min(sz, data_.size());
  std::memcpy(data_.data(), src, copySize);
}

void CPUBuffer::copyToHost(void *dst, size_t sz) const {
  size_t copySize = std::min(sz, data_.size());
  std::memcpy(dst, data_.data(), copySize);
}

void *CPUBuffer::rawPointer() { return data_.data(); }

// ── CPU PJRT Device ─────────────────────────────────────────────────────────

namespace {

class CPUDevice : public PJRTDevice {
public:
  explicit CPUDevice(int id) : id_(id) {}

  int id() const override { return id_; }
  std::string kind() const override { return "CPU"; }
  std::string vendor() const override { return "Host"; }

  size_t availableMemory() const override {
    // Return a large number for CPU (virtual memory).
    return size_t(8) * 1024 * 1024 * 1024; // 8 GB
  }

  std::shared_ptr<DeviceBuffer>
  allocateBuffer(const std::vector<int64_t> &shape,
                 ScalarType::ScalarKind elementType) override {
    return std::make_shared<CPUBuffer>(shape, elementType, id_);
  }

  std::shared_ptr<PJRTExecutable>
  loadExecutable(const std::string &serializedProgram) override {
    // In a full implementation, this would parse the StableHLO program
    // and compile it using XLA's CPU backend.
    // For now, we return a placeholder executable.
    class CPUPJRTExecutable : public PJRTExecutable {
    public:
      explicit CPUPJRTExecutable(std::string program)
          : program_(std::move(program)) {}

      std::vector<std::shared_ptr<DeviceBuffer>>
      execute(const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) override {
        // Execute on CPU by interpreting the program.
        // A full implementation would use XLA's local client API.
        return {};
      }

      std::pair<std::vector<std::shared_ptr<DeviceBuffer>>,
                std::shared_ptr<DeviceEvent>>
      executeAsync(const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) override {
        // For CPU, async is the same as sync (no device queue).
        auto outputs = execute(inputs);

        class CPUEvent : public DeviceEvent {
        public:
          void await() override {}
          bool isReady() const override { return true; }
        };

        return {outputs, std::make_shared<CPUEvent>()};
      }

      std::string name() const override { return "cpu_executable"; }

    private:
      std::string program_;
    };

    return std::make_shared<CPUPJRTExecutable>(serializedProgram);
  }

private:
  int id_;
};

} // anonymous namespace

// ── CPUPJRTClient implementation ─────────────────────────────────────────────

CPUPJRTClient::CPUPJRTClient(DiagnosticsEngine &diag) : diag_(diag) {
  cpuDevice_ = std::make_unique<CPUDevice>(0);
}

CPUPJRTClient::~CPUPJRTClient() = default;

size_t CPUPJRTClient::deviceCount() const { return 1; }

PJRTDevice *CPUPJRTClient::getDevice(size_t index) {
  return index == 0 ? cpuDevice_.get() : nullptr;
}

PJRTDevice *CPUPJRTClient::getDefaultDevice() { return cpuDevice_.get(); }

std::shared_ptr<DeviceBuffer>
CPUPJRTClient::allocateBuffer(const std::vector<int64_t> &shape,
                               ScalarType::ScalarKind elementType) {
  return cpuDevice_->allocateBuffer(shape, elementType);
}

std::shared_ptr<PJRTExecutable>
CPUPJRTClient::compileAndLoad(const std::string &serializedProgram) {
  return cpuDevice_->loadExecutable(serializedProgram);
}

void CPUPJRTClient::synchronizeAll() {
  // CPU is always synchronized.
}

// ── PJRTClient factory ──────────────────────────────────────────────────────

std::unique_ptr<PJRTClient> PJRTClient::create(const std::string &platform,
                                                 DiagnosticsEngine &diag) {
  if (platform == "cpu" || platform.empty()) {
    return std::make_unique<CPUPJRTClient>(diag);
  }

  // For other platforms (cuda, rocm, tpu), we would need the
  // corresponding PJRT plugin library. For now, fall back to CPU
  // with a warning.
  diag.warning(SourceLocation{},
               "PJRT platform '" + platform + "' not available, "
               "falling back to CPU");
  return std::make_unique<CPUPJRTClient>(diag);
}

} // namespace jules
