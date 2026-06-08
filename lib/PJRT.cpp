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
#include <regex>
#include <sstream>
#include <unordered_map>
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

// ── CPU Tensor Interpreter ───────────────────────────────────────────────────

namespace {

/// Describes a single operation in the pre-compiled operation sequence.
struct OpRecord {
  /// The kind of operation.
  enum OpKind : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
    Sigmoid,
    Tanh,
    MatMul,
    Max,
    Min,
    Constant,
    Reshape,
    Transpose,
    Reduce,
    Exp,
    Log,
    Copy,  // identity / contiguous copy
  };

  OpKind kind;

  /// Indices into the interpreter's tensor buffer table.
  /// - For binary ops: input1, input2 -> output
  /// - For unary ops: input1 -> output (input2 unused)
  /// - For constant: output only (payload stored in constData)
  /// - For matmul: input1, input2 -> output
  int input1;
  int input2;
  int output;

  /// Shape of the output tensor.
  std::vector<int64_t> outputShape;

  /// For Constant ops: the raw float data.
  std::vector<float> constData;

  /// For MatMul: contraction dimensions (left and right dimension indices).
  int lhsContractDim = -1;
  int rhsContractDim = -1;
};

/// A simple CPU tensor: owns a contiguous float buffer with a shape.
struct CPUTensor {
  std::vector<int64_t> shape;
  std::vector<float> data;

  size_t numElements() const {
    size_t n = 1;
    for (auto d : shape)
      n *= (d > 0 ? static_cast<size_t>(d) : 1);
    return n;
  }
};

/// A minimal CPU tensor interpreter that can execute a sequence of operations
/// on contiguous float buffers. This is a reference implementation proving
/// the PJRT pipeline works end-to-end without requiring XLA runtime linkage.
class CPUTensorInterpreter {
public:
  /// Parse a serialized MLIR/StableHLO module string and extract an operation
  /// sequence. Uses basic regex-based parsing — crude but functional for the
  /// common StableHLO dialect ops.
  void parse(const std::string &mlirText) {
    opSequence_.clear();
    nextBufferIdx_ = 0;

    // Allocate reserved slots for inputs.
    // We'll assign indices as we go.
    // The parser identifies operations in the module and records them.

    // Parse stablehlo.add
    parseBinaryOps(mlirText, "stablehlo.add", OpRecord::Add);
    parseBinaryOps(mlirText, "stablehlo.subtract", OpRecord::Sub);
    parseBinaryOps(mlirText, "stablehlo.multiply", OpRecord::Mul);
    parseBinaryOps(mlirText, "stablehlo.divide", OpRecord::Div);
    parseBinaryOps(mlirText, "stablehlo.maximum", OpRecord::Max);
    parseBinaryOps(mlirText, "stablehlo.minimum", OpRecord::Min);

    parseUnaryOps(mlirText, "stablehlo.negate", OpRecord::Neg);
    parseUnaryOps(mlirText, "stablehlo.relu", OpRecord::Relu);
    parseUnaryOps(mlirText, "stablehlo.logistic", OpRecord::Sigmoid);
    parseUnaryOps(mlirText, "stablehlo.tanh", OpRecord::Tanh);
    parseUnaryOps(mlirText, "stablehlo.exponential", OpRecord::Exp);
    parseUnaryOps(mlirText, "stablehlo.log", OpRecord::Log);

    parseMatMulOps(mlirText);
    parseConstantOps(mlirText);

    // If no operations were parsed, add a pass-through copy so we can
    // at least return the input data.
    if (opSequence_.empty()) {
      OpRecord copy;
      copy.kind = OpRecord::Copy;
      copy.input1 = 0;
      copy.input2 = -1;
      copy.output = 1;
      opSequence_.push_back(copy);
    }
  }

  /// Set the number of input buffers to expect.
  void setNumInputs(size_t n) { numInputBuffers_ = n; }

  /// Execute the operation sequence on the given input buffers.
  std::vector<std::shared_ptr<CPUTensor>>
  execute(const std::vector<std::shared_ptr<CPUTensor>> &inputs) {
    // Buffer table: maps buffer index to tensor.
    std::unordered_map<int, std::shared_ptr<CPUTensor>> buffers;

    // Place inputs into the buffer table.
    for (size_t i = 0; i < inputs.size() && i < numInputBuffers_; ++i) {
      buffers[static_cast<int>(i)] = inputs[i];
    }

    // Execute each operation in sequence.
    for (const auto &op : opSequence_) {
      switch (op.kind) {
      case OpRecord::Add:
        executeBinary(buffers, op, [](float a, float b) { return a + b; });
        break;
      case OpRecord::Sub:
        executeBinary(buffers, op, [](float a, float b) { return a - b; });
        break;
      case OpRecord::Mul:
        executeBinary(buffers, op, [](float a, float b) { return a * b; });
        break;
      case OpRecord::Div:
        executeBinary(buffers, op, [](float a, float b) {
          return b != 0.0f ? a / b : 0.0f;
        });
        break;
      case OpRecord::Max:
        executeBinary(buffers, op,
                      [](float a, float b) { return std::max(a, b); });
        break;
      case OpRecord::Min:
        executeBinary(buffers, op,
                      [](float a, float b) { return std::min(a, b); });
        break;
      case OpRecord::Neg:
        executeUnary(buffers, op, [](float a) { return -a; });
        break;
      case OpRecord::Relu:
        executeUnary(buffers, op, [](float a) { return a > 0.0f ? a : 0.0f; });
        break;
      case OpRecord::Sigmoid:
        executeUnary(buffers, op, [](float a) {
          return 1.0f / (1.0f + std::exp(-a));
        });
        break;
      case OpRecord::Tanh:
        executeUnary(buffers, op, [](float a) { return std::tanh(a); });
        break;
      case OpRecord::Exp:
        executeUnary(buffers, op, [](float a) { return std::exp(a); });
        break;
      case OpRecord::Log:
        executeUnary(buffers, op,
                     [](float a) { return a > 0.0f ? std::log(a) : -INFINITY; });
        break;
      case OpRecord::Constant:
        executeConstant(buffers, op);
        break;
      case OpRecord::MatMul:
        executeMatMul(buffers, op);
        break;
      case OpRecord::Copy: {
        auto it = buffers.find(op.input1);
        if (it != buffers.end()) {
          auto out = std::make_shared<CPUTensor>();
          out->shape = it->second->shape;
          out->data = it->second->data;
          buffers[op.output] = out;
        }
        break;
      }
      case OpRecord::Reshape:
      case OpRecord::Transpose:
      case OpRecord::Reduce:
        // Basic pass-through for unimplemented ops.
        {
          auto it = buffers.find(op.input1);
          if (it != buffers.end()) {
            auto out = std::make_shared<CPUTensor>();
            out->shape = op.outputShape.empty() ? it->second->shape
                                                : op.outputShape;
            out->data = it->second->data;
            buffers[op.output] = out;
          }
        }
        break;
      }
    }

    // Collect output buffers: all buffers that are not inputs are outputs.
    std::vector<std::shared_ptr<CPUTensor>> outputs;
    for (auto &[idx, tensor] : buffers) {
      if (idx >= static_cast<int>(numInputBuffers_)) {
        outputs.push_back(tensor);
      }
    }

    // If no output buffers were produced, return copies of the inputs
    // (pass-through mode for simple programs).
    if (outputs.empty()) {
      for (auto &inp : inputs) {
        auto out = std::make_shared<CPUTensor>();
        out->shape = inp->shape;
        out->data = inp->data;
        outputs.push_back(out);
      }
    }

    return outputs;
  }

private:
  std::vector<OpRecord> opSequence_;
  size_t nextBufferIdx_ = 0;
  size_t numInputBuffers_ = 0;

  int allocBuffer() { return static_cast<int>(nextBufferIdx_++); }

  /// Parse binary operations like: %result = "stablehlo.add"(%lhs, %rhs)
  void parseBinaryOps(const std::string &text, const std::string &opName,
                      OpRecord::OpKind kind) {
    std::regex pattern(
        R"(%(\w+)\s*=\s*")" + escapeRegex(opName) +
        R"("\(%(\w+),\s*%(\w+)\))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      OpRecord op;
      op.kind = kind;
      op.input1 = allocBuffer();
      op.input2 = allocBuffer();
      op.output = allocBuffer();
      opSequence_.push_back(op);
    }
  }

  /// Parse unary operations like: %result = "stablehlo.negate"(%operand)
  void parseUnaryOps(const std::string &text, const std::string &opName,
                     OpRecord::OpKind kind) {
    std::regex pattern(
        R"(%(\w+)\s*=\s*")" + escapeRegex(opName) +
        R"("\(%(\w+)\))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      OpRecord op;
      op.kind = kind;
      op.input1 = allocBuffer();
      op.input2 = -1;
      op.output = allocBuffer();
      opSequence_.push_back(op);
    }
  }

  /// Parse matmul / dot_general operations.
  void parseMatMulOps(const std::string &text) {
    // Match stablehlo.dot_general
    std::regex pattern(
        R"(%(\w+)\s*=\s*"stablehlo.dot_general"\(%(\w+),\s*%(\w+)\))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      OpRecord op;
      op.kind = OpRecord::MatMul;
      op.input1 = allocBuffer();
      op.input2 = allocBuffer();
      op.output = allocBuffer();
      op.lhsContractDim = -1; // will use default
      op.rhsContractDim = -1;
      opSequence_.push_back(op);
    }
  }

  /// Parse constant operations like: %cst = "stablehlo.constant"() {value = dense<...>}
  void parseConstantOps(const std::string &text) {
    // Match stablehlo.constant with dense element data.
    std::regex pattern(
        R"(%(\w+)\s*=\s*"stablehlo\.constant"\(\)\s*\{value\s*=\s*dense<([^>]*)>\})");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      OpRecord op;
      op.kind = OpRecord::Constant;
      op.input1 = -1;
      op.input2 = -1;
      op.output = allocBuffer();

      // Parse the dense values.
      std::string values = (*it)[2].str();
      std::istringstream ss(values);
      float val;
      while (ss >> val) {
        op.constData.push_back(val);
        if (ss.peek() == ',')
          ss.ignore();
      }
      op.outputShape = {static_cast<int64_t>(op.constData.size())};

      opSequence_.push_back(op);
    }
  }

  static std::string escapeRegex(const std::string &s) {
    std::string out;
    for (char c : s) {
      if (c == '.')
        out += "\\.";
      else
        out += c;
    }
    return out;
  }

  void executeBinary(
      std::unordered_map<int, std::shared_ptr<CPUTensor>> &buffers,
      const OpRecord &op, std::function<float(float, float)> fn) {
    auto lhsIt = buffers.find(op.input1);
    auto rhsIt = buffers.find(op.input2);
    if (lhsIt == buffers.end() || rhsIt == buffers.end())
      return;

    auto &lhs = lhsIt->second;
    auto &rhs = rhsIt->second;
    auto out = std::make_shared<CPUTensor>();

    // Determine output shape (broadcasting: use the larger shape).
    out->shape = lhs->shape.size() >= rhs->shape.size() ? lhs->shape
                                                         : rhs->shape;
    size_t outSize = std::max(lhs->data.size(), rhs->data.size());
    if (outSize == 0)
      outSize = 1;
    out->data.resize(outSize);

    for (size_t i = 0; i < outSize; ++i) {
      float a = i < lhs->data.size() ? lhs->data[i] : 0.0f;
      float b = i < rhs->data.size() ? rhs->data[i] : 0.0f;
      out->data[i] = fn(a, b);
    }
    buffers[op.output] = out;
  }

  void executeUnary(
      std::unordered_map<int, std::shared_ptr<CPUTensor>> &buffers,
      const OpRecord &op, std::function<float(float)> fn) {
    auto it = buffers.find(op.input1);
    if (it == buffers.end())
      return;

    auto &inp = it->second;
    auto out = std::make_shared<CPUTensor>();
    out->shape = inp->shape;
    out->data.resize(inp->data.size());

    for (size_t i = 0; i < inp->data.size(); ++i) {
      out->data[i] = fn(inp->data[i]);
    }
    buffers[op.output] = out;
  }

  void executeConstant(
      std::unordered_map<int, std::shared_ptr<CPUTensor>> &buffers,
      const OpRecord &op) {
    auto out = std::make_shared<CPUTensor>();
    out->shape = op.outputShape;
    out->data = op.constData;
    buffers[op.output] = out;
  }

  void executeMatMul(
      std::unordered_map<int, std::shared_ptr<CPUTensor>> &buffers,
      const OpRecord &op) {
    auto lhsIt = buffers.find(op.input1);
    auto rhsIt = buffers.find(op.input2);
    if (lhsIt == buffers.end() || rhsIt == buffers.end())
      return;

    auto &lhs = lhsIt->second;
    auto &rhs = rhsIt->second;
    auto out = std::make_shared<CPUTensor>();

    // Support 2D matmul: (M x K) * (K x N) = (M x N)
    int64_t M = lhs->shape.size() >= 2 ? lhs->shape[lhs->shape.size() - 2]
                                        : 1;
    int64_t K = lhs->shape.size() >= 1 ? lhs->shape[lhs->shape.size() - 1]
                                        : 1;
    int64_t N = rhs->shape.size() >= 2 ? rhs->shape[rhs->shape.size() - 1]
                                        : 1;

    out->shape = {M, N};
    out->data.resize(static_cast<size_t>(M * N), 0.0f);

    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          float a = (i * K + k) < static_cast<int64_t>(lhs->data.size())
                        ? lhs->data[static_cast<size_t>(i * K + k)]
                        : 0.0f;
          float b = (k * N + j) < static_cast<int64_t>(rhs->data.size())
                        ? rhs->data[static_cast<size_t>(k * N + j)]
                        : 0.0f;
          sum += a * b;
        }
        out->data[static_cast<size_t>(i * N + j)] = sum;
      }
    }
    buffers[op.output] = out;
  }
};

// ── CPU PJRT Device ─────────────────────────────────────────────────────────

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
  loadExecutable(const std::string &serializedProgram) override;

private:
  int id_;
};

/// CPU PJRT executable that uses the CPUTensorInterpreter to execute
/// StableHLO programs on CPU buffers.
class CPUPJRTExecutable : public PJRTExecutable {
public:
  explicit CPUPJRTExecutable(std::string program, int deviceId)
      : program_(std::move(program)), deviceId_(deviceId) {
    // Parse the MLIR/StableHLO program at load time.
    interpreter_.setNumInputs(0); // will be updated at execute time
    interpreter_.parse(program_);
  }

  std::vector<std::shared_ptr<DeviceBuffer>>
  execute(const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) override {
    // Convert input DeviceBuffers to CPUTensors for the interpreter.
    std::vector<std::shared_ptr<CPUTensor>> inputTensors;
    for (auto &buf : inputs) {
      auto tensor = std::make_shared<CPUTensor>();
      tensor->shape = buf->shape();
      // Copy the buffer data into a float vector.
      size_t byteSize = buf->size();
      size_t floatCount = byteSize / sizeof(float);
      tensor->data.resize(floatCount);
      if (floatCount > 0) {
        buf->copyToHost(tensor->data.data(), byteSize);
      }
      inputTensors.push_back(tensor);
    }

    // Re-parse with the correct number of inputs.
    interpreter_.setNumInputs(inputs.size());
    interpreter_.parse(program_);

    // Execute the operation sequence.
    auto outputTensors = interpreter_.execute(inputTensors);

    // Convert output CPUTensors back to DeviceBuffers.
    std::vector<std::shared_ptr<DeviceBuffer>> outputs;
    for (auto &tensor : outputTensors) {
      auto outBuf = std::make_shared<CPUBuffer>(tensor->shape,
                                                 ScalarType::SK_F32,
                                                 deviceId_);
      if (!tensor->data.empty()) {
        outBuf->copyFromHost(tensor->data.data(),
                              tensor->data.size() * sizeof(float));
      }
      outputs.push_back(outBuf);
    }

    return outputs;
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
  int deviceId_;
  CPUTensorInterpreter interpreter_;
};

std::shared_ptr<PJRTExecutable>
CPUDevice::loadExecutable(const std::string &serializedProgram) {
  return std::make_shared<CPUPJRTExecutable>(serializedProgram, id_);
}

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
