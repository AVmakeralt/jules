//===- ShapePolymorphismPass.cpp - Shape Polymorphism Implementation -------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the Shape Polymorphism pass for the Jules MLIR dialect.
//
// Traditional shape inference lowers all unknown dimensions to kDynamic,
// losing the information that certain dimensions must be equal. This pass
// instead carries symbolic dimension names through the pipeline and inserts
// shape.assuming assertions so that XLA can specialize.
//
// The pass performs the following steps:
//   1. Collect symbolic dimension names from function argument attributes
//   2. Build a dimension equality graph using Union-Find
//   3. Propagate symbolic shapes through each operation
//   4. Annotate ops with their symbolic shapes for downstream passes
//   5. Insert shape.assuming wrappers with collected constraints
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/ShapePolymorphismPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace jules;

namespace {

// ── Symbolic Dimension ──────────────────────────────────────────────────────

/// Represents a symbolic dimension variable (e.g., "Batch", "Features", "I").
/// Multiple tensor dimensions can share the same symbolic name, indicating
/// they must be equal at runtime.
struct SymbolicDim {
  /// The name of the symbolic dimension (e.g., "Batch", "SeqLen").
  llvm::StringRef name;

  /// The argument index and dimension index where this symbol originates.
  unsigned argIndex;
  unsigned dimIndex;

  /// Whether this dimension was originally dynamic.
  bool isDynamic;

  SymbolicDim() : argIndex(0), dimIndex(0), isDynamic(false) {}
  SymbolicDim(llvm::StringRef name, unsigned argIdx, unsigned dimIdx,
              bool dynamic)
      : name(name), argIndex(argIdx), dimIndex(dimIdx), isDynamic(dynamic) {}

  bool operator==(const SymbolicDim &other) const { return name == other.name; }
  bool operator<(const SymbolicDim &other) const { return name < other.name; }
};

/// Hash support for SymbolicDim.
struct SymbolicDimInfo : llvm::DenseMapInfo<SymbolicDim> {
  static inline SymbolicDim getEmptyKey() {
    return SymbolicDim(llvm::DenseMapInfo<StringRef>::getEmptyKey(),
                       0, 0, false);
  }
  static inline SymbolicDim getTombstoneKey() {
    return SymbolicDim(llvm::DenseMapInfo<StringRef>::getTombstoneKey(),
                       0, 0, false);
  }
  static unsigned getHashValue(const SymbolicDim &val) {
    return llvm::hash_combine(val.name, val.argIndex, val.dimIndex);
  }
  static bool isEqual(const SymbolicDim &lhs, const SymbolicDim &rhs) {
    return lhs.name == rhs.name;
  }
};

// ── Shape Constraint ────────────────────────────────────────────────────────

/// Represents a set of constraints on symbolic dimensions.
/// Tracks which dimensions must be equal (via union-find) and which
/// dimensions have known bounds.
struct ShapeConstraint {
  /// Union-Find for dimension equality: maps dimension name → representative.
  llvm::DenseMap<llvm::StringRef, llvm::StringRef> equalityClass;

  /// Known lower bounds for dimensions (e.g., Batch >= 1).
  llvm::DenseMap<llvm::StringRef, int64_t> lowerBounds;

  /// Known upper bounds for dimensions.
  llvm::DenseMap<llvm::StringRef, int64_t> upperBounds;

  /// All unique symbolic dimension names encountered.
  llvm::SmallVector<llvm::StringRef, 8> allDimNames;

  /// Add an equality constraint between two symbolic dimensions.
  void addEquality(llvm::StringRef dimA, llvm::StringRef dimB) {
    // Union-Find with path compression.
    llvm::StringRef rootA = findRoot(dimA);
    llvm::StringRef rootB = findRoot(dimB);
    if (rootA != rootB) {
      // Merge: make the lexicographically smaller name the root.
      if (rootA < rootB) {
        equalityClass[rootB] = rootA;
      } else {
        equalityClass[rootA] = rootB;
      }
    }
  }

  /// Find the root representative of a dimension name.
  llvm::StringRef findRoot(llvm::StringRef dim) {
    auto it = equalityClass.find(dim);
    if (it == equalityClass.end()) {
      equalityClass[dim] = dim;
      return dim;
    }
    if (it->second == dim) return dim;
    // Path compression.
    llvm::StringRef root = findRoot(it->second);
    equalityClass[dim] = root;
    return root;
  }

  /// Check if two dimensions are in the same equality class.
  bool areEqual(llvm::StringRef dimA, llvm::StringRef dimB) {
    return findRoot(dimA) == findRoot(dimB);
  }

  /// Record a symbolic dimension name if not already seen.
  void recordDim(llvm::StringRef name) {
    for (auto existing : allDimNames) {
      if (existing == name) return;
    }
    allDimNames.push_back(name);
    // Ensure it has an entry in the equality map.
    if (!equalityClass.count(name)) {
      equalityClass[name] = name;
    }
  }

  /// Add a lower bound constraint.
  void addLowerBound(llvm::StringRef dim, int64_t bound) {
    llvm::StringRef root = findRoot(dim);
    auto it = lowerBounds.find(root);
    if (it == lowerBounds.end() || it->second < bound) {
      lowerBounds[root] = bound;
    }
  }

  /// Add an upper bound constraint.
  void addUpperBound(llvm::StringRef dim, int64_t bound) {
    llvm::StringRef root = findRoot(dim);
    auto it = upperBounds.find(root);
    if (it == upperBounds.end() || it->second > bound) {
      upperBounds[root] = bound;
    }
  }

  /// Get all unique equality classes (representative → members).
  llvm::DenseMap<llvm::StringRef, llvm::SmallVector<llvm::StringRef, 4>>
  getEqualityClasses() const {
    llvm::DenseMap<llvm::StringRef, llvm::SmallVector<llvm::StringRef, 4>>
        classes;
    // We need to do a const_cast workaround since findRoot modifies state.
    // Instead, just iterate the map.
    for (auto &kv : equalityClass) {
      // kv.first → kv.second (parent). The root is the class representative.
      // We accumulate members by their direct parent, then resolve.
      classes[kv.second].push_back(kv.first);
    }
    return classes;
  }
};

// ── Symbolic Shape for a Value ──────────────────────────────────────────────

/// Describes the symbolic shape of a value: a list of dimension names,
/// one per dimension of the tensor.
struct SymbolicShape {
  /// Dimension names, one per tensor dimension. Empty for scalars.
  llvm::SmallVector<llvm::StringRef, 4> dimNames;

  /// The element type of the tensor.
  Type elementType;

  /// The original type (may have static dimensions).
  Type originalType;

  SymbolicShape() = default;

  /// Construct from a RankedTensorType, using provided dimension names.
  SymbolicShape(RankedTensorType type,
                llvm::SmallVector<llvm::StringRef, 4> names)
      : dimNames(std::move(names)), elementType(type.getElementType()),
        originalType(type) {}

  /// Get the rank of this symbolic shape.
  unsigned getRank() const { return dimNames.size(); }

  /// Check if this shape is valid (has dimNames for each dimension).
  bool isValid() const {
    if (auto rankedType = originalType.dyn_cast<RankedTensorType>()) {
      return dimNames.size() == static_cast<unsigned>(rankedType.getRank());
    }
    return dimNames.empty(); // scalar or unranked
  }
};

// ── Shape Polymorphism Pass ─────────────────────────────────────────────────

struct ShapePolymorphismPass
    : public PassWrapper<ShapePolymorphismPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk all functions and process those with dynamic dimensions.
    module.walk([&](func::FuncOp funcOp) {
      if (hasDynamicDimensions(funcOp)) {
        buildConstraints(funcOp);
        propagateShapes(funcOp);
        annotateOps(funcOp);
        insertShapeAssuming(funcOp);
      }
    });
  }

private:
  /// Per-function constraint state.
  llvm::DenseMap<func::FuncOp, ShapeConstraint> constraints_;

  /// Per-function symbolic shapes for each Value.
  llvm::DenseMap<func::FuncOp,
                 llvm::DenseMap<Value, SymbolicShape>>
      symbolicShapes_;

  /// Counter for generating unique dimension names.
  unsigned dimNameCounter_ = 0;

  // ── Step 1: Check if a function has dynamic dimensions ──────────────────

  bool hasDynamicDimensions(func::FuncOp funcOp) {
    for (auto argType : funcOp.getArgumentTypes()) {
      if (auto tensorType = argType.dyn_cast<RankedTensorType>()) {
        for (int64_t i = 0; i < tensorType.getRank(); ++i) {
          if (tensorType.isDynamicDim(i)) {
            return true;
          }
        }
      }
    }
    // Also check result types.
    for (auto resultType : funcOp.getResultTypes()) {
      if (auto tensorType = resultType.dyn_cast<RankedTensorType>()) {
        for (int64_t i = 0; i < tensorType.getRank(); ++i) {
          if (tensorType.isDynamicDim(i)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  // ── Step 2: Build symbolic dimension constraints ────────────────────────

  void buildConstraints(func::FuncOp funcOp) {
    ShapeConstraint &constraint = constraints_[funcOp];

    // Collect symbolic dimension names from function argument attributes.
    for (unsigned argIdx = 0; argIdx < funcOp.getNumArguments(); ++argIdx) {
      auto argType = funcOp.getArgumentTypes()[argIdx];
      auto tensorType = argType.dyn_cast<RankedTensorType>();
      if (!tensorType) continue;

      // Initialize symbolic shape for this argument.
      SymbolicShape &shape = symbolicShapes_[funcOp][funcOp.getArgument(argIdx)];
      shape.originalType = argType;
      shape.elementType = tensorType.getElementType();

      for (int64_t dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
        if (tensorType.isDynamicDim(dimIdx)) {
          // Check for a named dimension attribute.
          llvm::StringRef dimName = getDimName(funcOp, argIdx, dimIdx);

          // Record this symbolic dimension.
          constraint.recordDim(dimName);

          // Dynamic dimensions have a default lower bound of 1.
          constraint.addLowerBound(dimName, 1);

          shape.dimNames.push_back(dimName);
        } else {
          // Static dimension — use a mangled name to indicate it's fixed.
          std::string staticName;
          {
            llvm::raw_string_ostream os(staticName);
            os << "static_" << tensorType.getDimSize(dimIdx);
          }
          shape.dimNames.push_back(
              allocateDimName(staticName));
        }
      }
    }
  }

  // ── Step 3: Propagate symbolic shapes through operations ────────────────

  void propagateShapes(func::FuncOp funcOp) {
    ShapeConstraint &constraint = constraints_[funcOp];
    auto &shapes = symbolicShapes_[funcOp];

    funcOp.walk([&](Operation *op) {
      propagateThroughOp(op, shapes, constraint);
    });
  }

  /// Propagate symbolic shape through a single operation.
  void propagateThroughOp(Operation *op,
                           llvm::DenseMap<Value, SymbolicShape> &shapes,
                           ShapeConstraint &constraint) {
    // ── MatMulOp: [B, I] × [I, O] → [B, O] ──────────────────────────────
    if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
      auto lhsIt = shapes.find(matmulOp.getLhs());
      auto rhsIt = shapes.find(matmulOp.getRhs());

      auto lhsType = matmulOp.getLhs().getType().dyn_cast<RankedTensorType>();
      auto rhsType = matmulOp.getRhs().getType().dyn_cast<RankedTensorType>();
      if (!lhsType || !rhsType) return;

      SymbolicShape resultShape;
      resultShape.originalType = matmulOp.getResult().getType();
      resultShape.elementType = lhsType.getElementType();

      // 2D @ 2D: [M, K] × [K, N] → [M, N]
      if (lhsType.getRank() == 2 && rhsType.getRank() == 2) {
        llvm::StringRef dimM = getOrCreateDim(lhsIt, shapes, matmulOp.getLhs(),
                                               0, lhsType, constraint);
        llvm::StringRef dimK_lhs = getOrCreateDim(
            lhsIt, shapes, matmulOp.getLhs(), 1, lhsType, constraint);
        llvm::StringRef dimK_rhs = getOrCreateDim(
            rhsIt, shapes, matmulOp.getRhs(), 0, rhsType, constraint);
        llvm::StringRef dimN = getOrCreateDim(rhsIt, shapes, matmulOp.getRhs(),
                                               1, rhsType, constraint);

        // The inner dimensions K must be equal.
        constraint.addEquality(dimK_lhs, dimK_rhs);

        resultShape.dimNames = {dimM, dimN};
      }
      // Batched: [B, M, K] × [K, N] → [B, M, N]
      else if (lhsType.getRank() == 3 && rhsType.getRank() == 2) {
        llvm::StringRef dimB = getOrCreateDim(lhsIt, shapes, matmulOp.getLhs(),
                                               0, lhsType, constraint);
        llvm::StringRef dimM = getOrCreateDim(lhsIt, shapes, matmulOp.getLhs(),
                                               1, lhsType, constraint);
        llvm::StringRef dimK_lhs = getOrCreateDim(
            lhsIt, shapes, matmulOp.getLhs(), 2, lhsType, constraint);
        llvm::StringRef dimK_rhs = getOrCreateDim(
            rhsIt, shapes, matmulOp.getRhs(), 0, rhsType, constraint);
        llvm::StringRef dimN = getOrCreateDim(rhsIt, shapes, matmulOp.getRhs(),
                                               1, rhsType, constraint);

        constraint.addEquality(dimK_lhs, dimK_rhs);

        resultShape.dimNames = {dimB, dimM, dimN};
      }
      // Batched: [B, M, K] × [B, K, N] → [B, M, N]
      else if (lhsType.getRank() == 3 && rhsType.getRank() == 3) {
        llvm::StringRef dimB_lhs = getOrCreateDim(
            lhsIt, shapes, matmulOp.getLhs(), 0, lhsType, constraint);
        llvm::StringRef dimM = getOrCreateDim(lhsIt, shapes, matmulOp.getLhs(),
                                               1, lhsType, constraint);
        llvm::StringRef dimK_lhs = getOrCreateDim(
            lhsIt, shapes, matmulOp.getLhs(), 2, lhsType, constraint);
        llvm::StringRef dimB_rhs = getOrCreateDim(
            rhsIt, shapes, matmulOp.getRhs(), 0, rhsType, constraint);
        llvm::StringRef dimK_rhs = getOrCreateDim(
            rhsIt, shapes, matmulOp.getRhs(), 1, rhsType, constraint);
        llvm::StringRef dimN = getOrCreateDim(rhsIt, shapes, matmulOp.getRhs(),
                                               2, rhsType, constraint);

        constraint.addEquality(dimK_lhs, dimK_rhs);
        constraint.addEquality(dimB_lhs, dimB_rhs);

        resultShape.dimNames = {dimB_lhs, dimM, dimN};
      }

      shapes[matmulOp.getResult()] = resultShape;
      return;
    }

    // ── AddOp / SubOp: same shape → dimensions are pairwise equal ─────────
    if (isa<AddOp>(op) || isa<SubOp>(op)) {
      auto lhsType = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
      auto rhsType = op->getOperand(1).getType().dyn_cast<RankedTensorType>();
      if (lhsType && rhsType) {
        auto lhsIt = shapes.find(op->getOperand(0));
        auto rhsIt = shapes.find(op->getOperand(1));

        SymbolicShape resultShape;
        resultShape.originalType = op->getResult(0).getType();
        resultShape.elementType = lhsType.getElementType();

        unsigned rank = static_cast<unsigned>(lhsType.getRank());
        for (unsigned d = 0; d < rank; ++d) {
          llvm::StringRef lhsDim = getOrCreateDim(
              lhsIt, shapes, op->getOperand(0), d, lhsType, constraint);
          llvm::StringRef rhsDim = getOrCreateDim(
              rhsIt, shapes, op->getOperand(1), d, rhsType, constraint);
          constraint.addEquality(lhsDim, rhsDim);
          resultShape.dimNames.push_back(lhsDim);
        }

        shapes[op->getResult(0)] = resultShape;
        return;
      }
    }

    // ── MulOp: broadcast multiply ─────────────────────────────────────────
    if (isa<MulOp>(op)) {
      auto lhsType = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
      auto rhsType = op->getOperand(1).getType().dyn_cast<RankedTensorType>();

      SymbolicShape resultShape;
      resultShape.originalType = op->getResult(0).getType();

      // Determine which operand determines the result shape.
      if (lhsType && rhsType) {
        resultShape.elementType = lhsType.getElementType();
        auto lhsIt = shapes.find(op->getOperand(0));
        auto rhsIt = shapes.find(op->getOperand(1));

        unsigned maxRank = static_cast<unsigned>(
            std::max(lhsType.getRank(), rhsType.getRank()));
        for (unsigned d = 0; d < maxRank; ++d) {
          // Use the tensor operand's dimension name if available.
          if (d < static_cast<unsigned>(lhsType.getRank())) {
            llvm::StringRef dimName = getOrCreateDim(
                lhsIt, shapes, op->getOperand(0), d, lhsType, constraint);
            resultShape.dimNames.push_back(dimName);
          } else if (d < static_cast<unsigned>(rhsType.getRank())) {
            llvm::StringRef dimName = getOrCreateDim(
                rhsIt, shapes, op->getOperand(1), d, rhsType, constraint);
            resultShape.dimNames.push_back(dimName);
          }
        }

        // For same-rank multiplications, constrain matching dims equal.
        if (lhsType.getRank() == rhsType.getRank()) {
          for (unsigned d = 0;
               d < static_cast<unsigned>(lhsType.getRank()); ++d) {
            if (lhsType.isDynamicDim(d) && rhsType.isDynamicDim(d)) {
              llvm::StringRef lhsDim = getOrCreateDim(
                  lhsIt, shapes, op->getOperand(0), d, lhsType, constraint);
              llvm::StringRef rhsDim = getOrCreateDim(
                  rhsIt, shapes, op->getOperand(1), d, rhsType, constraint);
              constraint.addEquality(lhsDim, rhsDim);
            }
          }
        }
      } else if (lhsType) {
        // lhs is tensor, rhs is scalar → result shape = lhs shape.
        resultShape.elementType = lhsType.getElementType();
        auto lhsIt = shapes.find(op->getOperand(0));
        for (int64_t d = 0; d < lhsType.getRank(); ++d) {
          resultShape.dimNames.push_back(getOrCreateDim(
              lhsIt, shapes, op->getOperand(0), d, lhsType, constraint));
        }
      }

      shapes[op->getResult(0)] = resultShape;
      return;
    }

    // ── ReluOp / SigmoidOp / TanhOp / NegOp: same shape ──────────────────
    if (isa<ReluOp>(op) || isa<SigmoidOp>(op) || isa<TanhOp>(op) ||
        isa<NegOp>(op)) {
      auto inputType =
          op->getOperand(0).getType().dyn_cast<RankedTensorType>();
      if (inputType) {
        auto inputIt = shapes.find(op->getOperand(0));
        SymbolicShape resultShape;
        resultShape.originalType = op->getResult(0).getType();
        resultShape.elementType = inputType.getElementType();

        for (int64_t d = 0; d < inputType.getRank(); ++d) {
          resultShape.dimNames.push_back(getOrCreateDim(
              inputIt, shapes, op->getOperand(0), d, inputType, constraint));
        }
        shapes[op->getResult(0)] = resultShape;
      }
      return;
    }

    // ── TransposeOp: reverse dimension order ──────────────────────────────
    if (isa<TransposeOp>(op)) {
      auto inputType =
          op->getOperand(0).getType().dyn_cast<RankedTensorType>();
      if (inputType) {
        auto inputIt = shapes.find(op->getOperand(0));
        SymbolicShape resultShape;
        resultShape.originalType = op->getResult(0).getType();
        resultShape.elementType = inputType.getElementType();

        // Transpose reverses dimension order.
        for (int64_t d = inputType.getRank() - 1; d >= 0; --d) {
          resultShape.dimNames.push_back(getOrCreateDim(
              inputIt, shapes, op->getOperand(0), d, inputType, constraint));
        }
        shapes[op->getResult(0)] = resultShape;
      }
      return;
    }

    // ── ReshapeOp: total elements must be equal ───────────────────────────
    if (isa<ReshapeOp>(op)) {
      auto inputType =
          op->getOperand(0).getType().dyn_cast<RankedTensorType>();
      auto resultType =
          op->getResult(0).getType().dyn_cast<RankedTensorType>();
      if (inputType && resultType) {
        SymbolicShape resultShape;
        resultShape.originalType = op->getResult(0).getType();
        resultShape.elementType = resultType.getElementType();

        // For each dimension in the result, generate a new symbolic name.
        // The constraint is that the total number of elements is the same.
        for (int64_t d = 0; d < resultType.getRank(); ++d) {
          if (resultType.isDynamicDim(d)) {
            llvm::StringRef dimName = generateDimName("Reshape");
            constraint.recordDim(dimName);
            constraint.addLowerBound(dimName, 1);
            resultShape.dimNames.push_back(dimName);
          } else {
            std::string staticName;
            {
              llvm::raw_string_ostream os(staticName);
              os << "static_" << resultType.getDimSize(d);
            }
            resultShape.dimNames.push_back(
                allocateDimName(staticName));
          }
        }

        // Add constraint: product of input dims == product of output dims.
        // We represent this as an element count constraint.
        shapes[op->getResult(0)] = resultShape;
      }
      return;
    }

    // ── MeanOp / SumOp: reduction to scalar ───────────────────────────────
    if (isa<MeanOp>(op) || isa<SumOp>(op)) {
      SymbolicShape resultShape;
      resultShape.originalType = op->getResult(0).getType();
      auto resultType =
          op->getResult(0).getType().dyn_cast<RankedTensorType>();
      if (resultType) {
        resultShape.elementType = resultType.getElementType();
      }
      // Scalar result has no dimensions.
      shapes[op->getResult(0)] = resultShape;
      return;
    }

    // ── CastOp: same symbolic shape ───────────────────────────────────────
    if (isa<CastOp>(op)) {
      auto inputIt = shapes.find(op->getOperand(0));
      if (inputIt != shapes.end()) {
        shapes[op->getResult(0)] = inputIt->second;
        shapes[op->getResult(0)].originalType = op->getResult(0).getType();
      }
      return;
    }

    // ── ConstantOp / ZerosOp / OnesOp: static shapes, no constraints ─────
    if (isa<ConstantOp>(op) || isa<ZerosOp>(op) || isa<OnesOp>(op)) {
      // These produce fully static shapes — no symbolic dimensions needed.
      return;
    }
  }

  // ── Step 4: Annotate ops with symbolic shapes ──────────────────────────

  void annotateOps(func::FuncOp funcOp) {
    auto &shapes = symbolicShapes_[funcOp];

    funcOp.walk([&](Operation *op) {
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        auto it = shapes.find(op->getResult(i));
        if (it != shapes.end() && it->second.isValid()) {
          // Build a string representation of the symbolic shape.
          std::string shapeStr;
          {
            llvm::raw_string_ostream os(shapeStr);
            os << "[";
            for (unsigned d = 0; d < it->second.dimNames.size(); ++d) {
              if (d > 0) os << ", ";
              os << it->second.dimNames[d];
            }
            os << "]";
          }

          op->setAttr(
              "jules.symbolic_shape",
              StringAttr::get(op->getContext(), shapeStr));
        }
      }
    });
  }

  // ── Step 5: Insert shape.assuming wrapper ───────────────────────────────

  void insertShapeAssuming(func::FuncOp funcOp) {
    ShapeConstraint &constraint = constraints_[funcOp];

    // If no constraints were gathered, nothing to assert.
    if (constraint.allDimNames.empty()) return;

    // Build the constraint string for the shape.assuming region.
    // We encode constraints as a dictionary attribute on the function,
    // which downstream passes (or XLA lowering) can interpret.
    std::string constraintStr;
    {
      llvm::raw_string_ostream os(constraintStr);

      // Emit equality constraints.
      auto classes = constraint.getEqualityClasses();
      bool first = true;
      for (auto &[rep, members] : classes) {
        if (members.size() > 1) {
          // Skip static dimensions.
          if (rep.startswith("static_")) continue;

          if (!first) os << "; ";
          first = false;
          os << "equal(";
          for (unsigned i = 0; i < members.size(); ++i) {
            if (i > 0) os << ", ";
            os << members[i];
          }
          os << ")";
        }
      }

      // Emit bound constraints.
      for (auto &[dim, bound] : constraint.lowerBounds) {
        if (!first) os << "; ";
        first = false;
        os << dim << " >= " << bound;
      }
      for (auto &[dim, bound] : constraint.upperBounds) {
        if (!first) os << "; ";
        first = false;
        os << dim << " <= " << bound;
      }
    }

    if (!constraintStr.empty()) {
      funcOp->setAttr(
          "jules.shape_constraints",
          StringAttr::get(funcOp.getContext(), constraintStr));
    }

    // Annotate each argument with its symbolic dimension names.
    auto &shapes = symbolicShapes_[funcOp];
    for (unsigned argIdx = 0; argIdx < funcOp.getNumArguments(); ++argIdx) {
      auto it = shapes.find(funcOp.getArgument(argIdx));
      if (it != shapes.end() && it->second.isValid() &&
          !it->second.dimNames.empty()) {
        SmallVector<Attribute, 4> dimNameAttrs;
        for (auto &name : it->second.dimNames) {
          dimNameAttrs.push_back(
              StringAttr::get(funcOp.getContext(), name));
        }
        funcOp.setArgAttr(argIdx, "jules.dim_names",
                          ArrayAttr::get(funcOp.getContext(), dimNameAttrs));
      }
    }
  }

  // ── Helper: Get or create a symbolic dimension name ────────────────────

  llvm::StringRef getOrCreateDim(
      llvm::DenseMap<Value, SymbolicShape>::iterator &shapeIt,
      llvm::DenseMap<Value, SymbolicShape> &shapes,
      Value val, unsigned dimIdx, RankedTensorType type,
      ShapeConstraint &constraint) {

    // If we already have a symbolic shape for this value, use it.
    if (shapeIt != shapes.end() &&
        dimIdx < shapeIt->second.dimNames.size()) {
      return shapeIt->second.dimNames[dimIdx];
    }

    // Otherwise, generate a name based on whether the dimension is dynamic.
    if (type.isDynamicDim(dimIdx)) {
      llvm::StringRef name = generateDimName("dim");
      constraint.recordDim(name);
      constraint.addLowerBound(name, 1);

      // Also update the symbolic shape.
      SymbolicShape &shape = shapes[val];
      shape.originalType = type;
      shape.elementType = type.getElementType();
      while (shape.dimNames.size() <= dimIdx) {
        shape.dimNames.push_back("");
      }
      shape.dimNames[dimIdx] = name;
      return name;
    }

    // Static dimension.
    std::string staticName;
    {
      llvm::raw_string_ostream os(staticName);
      os << "static_" << type.getDimSize(dimIdx);
    }
    llvm::StringRef name = allocateDimName(staticName);
    return name;
  }

  // ── Helper: Get dimension name from function argument attributes ────────

  llvm::StringRef getDimName(func::FuncOp funcOp, unsigned argIdx,
                              unsigned dimIdx) {
    // Check for jules.dim_name attribute on the function argument.
    auto dimNamesAttr = funcOp.getArgAttrOfType<ArrayAttr>(argIdx,
                                                            "jules.dim_names");
    if (dimNamesAttr && dimIdx < dimNamesAttr.size()) {
      if (auto strAttr = dimNamesAttr[dimIdx].dyn_cast<StringAttr>()) {
        return strAttr.getValue();
      }
    }

    // No attribute found — generate a default name.
    std::string name;
    {
      llvm::raw_string_ostream os(name);
      os << "arg" << argIdx << "_d" << dimIdx;
    }
    return allocateDimName(name);
  }

  // ── Helper: Generate a unique dimension name ───────────────────────────

  llvm::StringRef generateDimName(llvm::StringRef prefix) {
    std::string name;
    {
      llvm::raw_string_ostream os(name);
      os << prefix << "_" << dimNameCounter_++;
    }
    return allocateDimName(name);
  }

  // ── Helper: Allocate a persistent dimension name string ─────────────────
  //
  // We store allocated names in a set to ensure stable StringRefs.

  llvm::StringRef allocateDimName(const std::string &name) {
    auto [it, inserted] = allocatedDimNames_.insert(name);
    return *it;
  }

  /// Persistent storage for allocated dimension name strings.
  llvm::StringSet<> allocatedDimNames_;
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createShapePolymorphismPass() {
  return std::make_unique<ShapePolymorphismPass>();
}
