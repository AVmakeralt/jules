//===- PJRT.h - PJRT Device API Integration --------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PJRT (Predictive Just-In-Time Runtime) device API
// integration for the Jules compiler. PJRT is OpenXLA's uniform device
// C/C++ API used by frameworks like JAX.
//
// The PJRT integration provides:
//   - Device discovery and management (CPU, GPU, TPU)
//   - Buffer allocation and data transfer
//   - Executable loading and execution
//   - Async execution with event synchronization
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PJRT_H
#define JULES_PJRT_H

#include "jules/Diagnostics.h"
#include "jules/AST.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace jules {

// ── Device Buffer ───────────────────────────────────────────────────────────

/// A buffer allocated on a device. Owns the device memory.
class DeviceBuffer {
public:
  virtual ~DeviceBuffer() = default;

  /// Get the size of the buffer in bytes.
  virtual size_t size() const = 0;

  /// Get the shape of the data in this buffer.
  virtual std::vector<int64_t> shape() const = 0;

  /// Get the element type.
  virtual ScalarType::ScalarKind elementType() const = 0;

  /// Copy data from host to this device buffer.
  virtual void copyFromHost(const void *data, size_t size) = 0;

  /// Copy data from this device buffer to host.
  virtual void copyToHost(void *data, size_t size) const = 0;

  /// Get a raw pointer to the device memory (device-side only).
  virtual void *rawPointer() = 0;

  /// Get the device ID this buffer is allocated on.
  virtual int deviceId() const = 0;

  /// Is this buffer on the host (CPU)?
  virtual bool isHostBuffer() const = 0;
};

// ── Device Event ────────────────────────────────────────────────────────────

/// An event for synchronizing async device operations.
class DeviceEvent {
public:
  virtual ~DeviceEvent() = default;

  /// Block until this event is complete.
  virtual void await() = 0;

  /// Check if this event is complete without blocking.
  virtual bool isReady() const = 0;
};

// ── PJRT Executable ─────────────────────────────────────────────────────────

/// A compiled executable loaded onto a device via PJRT.
class PJRTExecutable {
public:
  virtual ~PJRTExecutable() = default;

  /// Execute the program with the given input buffers.
  /// Returns output buffers.
  virtual std::vector<std::shared_ptr<DeviceBuffer>>
  execute(const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) = 0;

  /// Execute asynchronously, returning an event to wait on.
  virtual std::pair<std::vector<std::shared_ptr<DeviceBuffer>>,
                    std::shared_ptr<DeviceEvent>>
  executeAsync(const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) = 0;

  /// Get the name of this executable (for debugging).
  virtual std::string name() const = 0;
};

// ── PJRT Device ─────────────────────────────────────────────────────────────

/// A single device accessible via PJRT.
class PJRTDevice {
public:
  virtual ~PJRTDevice() = default;

  /// Get the device ID.
  virtual int id() const = 0;

  /// Get the device kind (e.g., "CPU", "NVIDIA_GPU", "TPU").
  virtual std::string kind() const = 0;

  /// Get the device vendor.
  virtual std::string vendor() const = 0;

  /// Get available memory in bytes.
  virtual size_t availableMemory() const = 0;

  /// Allocate a buffer on this device.
  virtual std::shared_ptr<DeviceBuffer>
  allocateBuffer(const std::vector<int64_t> &shape,
                 ScalarType::ScalarKind elementType) = 0;

  /// Load a compiled executable onto this device.
  /// The program is the serialized StableHLO/MLIR representation.
  virtual std::shared_ptr<PJRTExecutable>
  loadExecutable(const std::string &serializedProgram) = 0;
};

// ── PJRT Client ─────────────────────────────────────────────────────────────

/// The top-level PJRT client: manages devices and provides the API.
class PJRTClient {
public:
  virtual ~PJRTClient() = default;

  /// Initialize the PJRT client for the given platform.
  static std::unique_ptr<PJRTClient> create(const std::string &platform,
                                              DiagnosticsEngine &diag);

  /// Get the platform name (e.g., "cpu", "cuda", "rocm", "tpu").
  virtual std::string platform() const = 0;

  /// Get the number of available devices.
  virtual size_t deviceCount() const = 0;

  /// Get a device by index.
  virtual PJRTDevice *getDevice(size_t index) = 0;

  /// Get the default device.
  virtual PJRTDevice *getDefaultDevice() = 0;

  /// Allocate a buffer on the default device.
  virtual std::shared_ptr<DeviceBuffer>
  allocateBuffer(const std::vector<int64_t> &shape,
                 ScalarType::ScalarKind elementType) = 0;

  /// Compile and load a program onto the default device.
  virtual std::shared_ptr<PJRTExecutable>
  compileAndLoad(const std::string &serializedProgram) = 0;

  /// Synchronize all devices.
  virtual void synchronizeAll() = 0;
};

// ── CPU PJRT Implementation ─────────────────────────────────────────────────

/// A CPU-based PJRT client for development and testing.
class CPUPJRTClient : public PJRTClient {
public:
  explicit CPUPJRTClient(DiagnosticsEngine &diag);
  ~CPUPJRTClient() override;

  std::string platform() const override { return "cpu"; }
  size_t deviceCount() const override;
  PJRTDevice *getDevice(size_t index) override;
  PJRTDevice *getDefaultDevice() override;
  std::shared_ptr<DeviceBuffer>
  allocateBuffer(const std::vector<int64_t> &shape,
                 ScalarType::ScalarKind elementType) override;
  std::shared_ptr<PJRTExecutable>
  compileAndLoad(const std::string &serializedProgram) override;
  void synchronizeAll() override;

private:
  DiagnosticsEngine &diag_;
  std::unique_ptr<PJRTDevice> cpuDevice_;
};

// ── CPU Device Buffer ───────────────────────────────────────────────────────

/// A host (CPU) memory buffer.
class CPUBuffer : public DeviceBuffer {
public:
  CPUBuffer(std::vector<int64_t> shape, ScalarType::ScalarKind elementType,
            int deviceId = 0);
  ~CPUBuffer() override;

  size_t size() const override;
  std::vector<int64_t> shape() const override { return shape_; }
  ScalarType::ScalarKind elementType() const override { return elementType_; }
  void copyFromHost(const void *data, size_t sz) override;
  void copyToHost(void *data, size_t sz) const override;
  void *rawPointer() override;
  int deviceId() const override { return deviceId_; }
  bool isHostBuffer() const override { return true; }

private:
  std::vector<int64_t>   shape_;
  ScalarType::ScalarKind elementType_;
  int                    deviceId_;
  std::vector<uint8_t>   data_;
};

} // namespace jules

#endif // JULES_PJRT_H
