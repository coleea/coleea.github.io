---
title: 08_cute_dsl-pipeline
published: 0001-01-01
description: ""
tags: []
category: Default
draft: true
---

# CuTeDSL Compilation Pipeline: Python Code � SASS ISA

This document provides a comprehensive analysis of the complete compilation flow in CuTeDSL, tracing the transformation from Python code to final SASS (Shader Assembly) instructions.

---

## Table of Contents

1. [Overview](#overview)
2. [Compilation Pipeline Stages](#compilation-pipeline-stages)
3. [Intermediate Representations](#intermediate-representations)
4. [Tools and Libraries](#tools-and-libraries)
5. [Code Flow Diagram](#code-flow-diagram)
6. [Key Files and Entry Points](#key-files-and-entry-points)
7. [Configuration Options](#configuration-options)
8. [Debugging and Inspection](#debugging-and-inspection)

---

## Overview

CuTeDSL uses a sophisticated **7-stage compilation pipeline** that transforms high-level Python code into GPU machine code. The pipeline leverages:

- **MLIR** (Multi-Level Intermediate Representation) for flexible IR transformations
- **NVVM** (NVIDIA Virtual Machine) for PTX generation
- **CUDA Toolkit** (ptxas) for final assembly and machine code generation

### High-Level Pipeline

```
Python Code (@cute.jit)
    �
[1] AST Preprocessing
    �
[2] MLIR IR Generation (CuTe Dialect)
    �
[3] MLIR Transformations (cute-to-nvvm)
    �
[4] PTX Generation (libNVVM)
    �
[5] CUBIN Compilation (ptxas)
    �
[6] SASS Machine Code (in CUBIN)
    �
[7] Runtime Execution (CUDA Driver)
```

---

## Compilation Pipeline Stages

### Stage 1: Python Code � Python AST

**Location:** `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py`

**Purpose:** Parse and optionally transform Python abstract syntax tree

**Two Compilation Modes:**

#### 1. Preprocessor Mode (Default: `@jit(preprocess=True)`)
- **Process:** AST rewriting + tracing
- **Advantages:**
  - Preserves Python control flow structure (`for`, `while`, `if`/`else`)
  - Enables loop optimizations (tiling, vectorization, GPU thread mapping)
  - Supports data-dependent control flow
  - No branch loss or forced loop unrolling
- **How it works:**
  - Analyzes function's AST before execution
  - Converts Python constructs to structured MLIR operations
  - Maintains semantic equivalence

#### 2. Tracing-Only Mode (`@jit(preprocess=False)`)
- **Process:** Pure tracing approach
- **How it works:**
  - Executes function once with proxy arguments
  - Records tensor operations in execution order
  - Generates straight-line IR
- **Limitations:**
  - Untaken branches disappear
  - Loops get flattened/unrolled
  - Data-dependent control flow frozen at trace time
- **Advantages:**
  - Faster compilation
  - Simpler for pure arithmetic kernels

**Key Classes:**
- `DSLPreprocessor` - Handles AST transformations
- `ASTRewriter` - Rewrites Python AST nodes

**Example:**
```python
@cute.jit(preprocess=True)  # Enable AST preprocessing
def my_kernel(A, B, C):
    for i in range(N):  # Preserved as structured loop
        if condition(i):  # Preserved as conditional
            C[i] = A[i] + B[i]
```

---

### Stage 2: Python AST � MLIR (CuTe Dialect)

**Location:**
- `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py`
- `python/CuTeDSL/cutlass/base_dsl/dsl.py`

**Purpose:** Convert Python DSL operations to MLIR intermediate representation

**Process:**

1. **Function Decoration:**
   ```python
   @cute.jit
   def my_kernel(...):
       # Python code with CuTe tensor operations
   ```

2. **DSL Object Creation:**
   - `CutlassBaseDSL` inherits from `BaseDSL`
   - Creates MLIR context and module
   - Initializes GPU module: `gpu.GPUModuleOp`

3. **IR Generation:**
   - Traces function execution with proxy arguments
   - Converts Python operations � MLIR operations
   - Generates typed MLIR values

**MLIR Dialects Used:**
- `cute` - CuTe tensor algebra operations
- `gpu` - GPU module, kernel launch operations
- `func` - Function definitions and calls
- `scf` - Structured Control Flow (for, while, if)
- `arith` - Arithmetic operations
- `vector` - Vector operations
- `cuda` - CUDA runtime operations
- `nvvm` - NVIDIA intrinsics

**Key Operations:**
```python
# Example: Tensor creation
tensor = cute.make_tensor(layout, ptr)

# Example: Thread indexing
thread_idx = gpu.thread_id(gpu.Dimension.x)

# Example: Shared memory allocation
smem = gpu.alloc_shared_memory(size, dtype)
```

**GPU Module Structure:**
```mlir
gpu.module @kernel_module {
  gpu.func @my_kernel(%arg0: ...) kernel {
    // MLIR operations
    gpu.return
  }
}
```

---

### Stage 3: MLIR Transformations (CuTe � NVVM)

**Location:** `python/CuTeDSL/cutlass/base_dsl/compiler.py:136-194`

**Purpose:** Lower high-level operations to NVVM intrinsics

**Entry Point: `Compiler.compile()` method**

```python
def compile(self, module, pipeline: str, cuda_toolkit: str = "",
            arch: str = "", enable_verifier=False):
    pm = self.passmanager.PassManager.parse(pipeline)
    pm.enable_verifier(enable_verifier)
    pm.run(module.operation)  # Execute transformation passes
```

**Pipeline Configuration:**

From `cutlass.py:260-278`:
```python
pipeline = f"builtin.module(cute-to-nvvm{{cubin-format=bin {compile_options.to_str()} \
             enable-cuda-dialect=true cuda-dialect-external-module=true}})"
```

**Main Pass: `cute-to-nvvm`**

This critical pass performs:

1. **Tensor Operation Lowering:**
   - CuTe tensor operations � GPU memory operations
   - Layout transformations � Address calculations
   - Collective operations � Warp/thread group operations

2. **Memory Hierarchy Mapping:**
   - Register allocation
   - Shared memory management
   - Global memory access patterns
   - Texture/surface memory

3. **GPU-Specific Operations:**
   - Thread/block/grid indexing
   - Warp operations (shuffle, vote, reduce)
   - Synchronization primitives (barriers, fences)
   - Asynchronous memory operations (TMA, cp.async)

4. **NVVM Intrinsic Generation:**
   - Thread indexing: `nvvm.read_ptx_sreg_tid_x/y/z`
   - Warp operations: `nvvm.shfl_sync`, `nvvm.vote_ballot_sync`
   - Barriers: `nvvm.barrier`, `nvvm.barrier_arrive`, `nvvm.barrier_wait`
   - Memory: `nvvm.cp_async_commit_group`, `nvvm.cp_async_wait_group`
   - Math: `nvvm.fma`, `nvvm.sqrt`, etc.

**Optimization Passes Applied:**

Based on `CompileOptions`:
- Dead code elimination
- Constant folding
- Loop optimization
- Memory coalescing
- Register pressure reduction

**Pass Options:**
- `opt-level` (0-3): Optimization aggressiveness
- `cubin-chip`: Target architecture (sm_90, sm_100, sm_101, sm_103, sm_120, etc.)
- `toolkitPath`: CUDA toolkit location for libNVVM
- `preserve-line-info`: Debug information
- `enable-assertions`: Runtime checks

**Example Transformation:**

```python
# Python/CuTe DSL
tensor[i] = A[i] + B[i]

# � After cute-to-nvvm �

# NVVM IR
%0 = nvvm.read_ptx_sreg_tid_x : i32
%1 = gpu.load %A[%0] : f32
%2 = gpu.load %B[%0] : f32
%3 = arith.addf %1, %2 : f32
gpu.store %3, %tensor[%0] : f32
```

---

### Stage 4: NVVM � PTX

**Location:** Handled by libNVVM (part of CUDA toolkit)

**Purpose:** Generate PTX (Parallel Thread Execution) assembly

**PTX Overview:**
- Virtual ISA for NVIDIA GPUs
- Human-readable assembly language
- Architecture-independent (within compute capability bounds)
- JIT-compiled to SASS at runtime or AOT by ptxas

**libNVVM Integration:**

Path: `{CUDA_TOOLKIT_PATH}/nvvm/lib/libnvvm.so` (Linux) or `libnvvm.dll` (Windows)

The MLIR NVVM dialect is directly lowered to PTX by libNVVM during the compilation pipeline.

**PTX Characteristics:**

1. **Instructions:**
   - Load/Store: `ld.global`, `st.shared`
   - Arithmetic: `add.f32`, `mul.f32`, `fma.f32`
   - Control flow: `bra`, `call`, `ret`
   - Synchronization: `bar.sync`, `bar.arrive`
   - Warp operations: `shfl.sync`, `vote.ballot.sync`

2. **Registers:**
   - Predicate: `%p0, %p1, ...`
   - Integer: `%r0, %r1, ...`
   - Float: `%f0, %f1, ...`
   - Double: `%d0, %d1, ...`

3. **Memory Spaces:**
   - `.reg` - Registers
   - `.shared` - Shared memory
   - `.global` - Global memory
   - `.param` - Parameter memory
   - `.const` - Constant memory

**Example PTX:**

```ptx
.version 8.5
.target sm_90
.address_size 64

.visible .entry my_kernel(
    .param .u64 ptr_A,
    .param .u64 ptr_B,
    .param .u64 ptr_C
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<8>;

    // Thread index
    mov.u32 %r1, %tid.x;

    // Load A[i]
    ld.param.u64 %rd1, [ptr_A];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];

    // Load B[i]
    ld.param.u64 %rd5, [ptr_B];
    cvta.to.global.u64 %rd6, %rd5;
    add.s64 %rd7, %rd6, %rd3;
    ld.global.f32 %f2, [%rd7];

    // C[i] = A[i] + B[i]
    add.f32 %f3, %f1, %f2;

    // Store C[i]
    ld.param.u64 %rd8, [ptr_C];
    cvta.to.global.u64 %rd9, %rd8;
    add.s64 %rd10, %rd9, %rd3;
    st.global.f32 [%rd10], %f3;

    ret;
}
```

**PTX Dumping:**

If `KeepPTX` option enabled:
- Output path: `{dump_dir}/{function_name}.{arch}.ptx`
- Example: `my_kernel.sm_90.ptx`

---

### Stage 5: PTX � CUBIN

**Location:** Uses `ptxas` assembler from CUDA toolkit

**Purpose:** Compile PTX to CUBIN binary containing SASS machine code

**ptxas Tool:**

Path: `{CUDA_TOOLKIT_PATH}/bin/ptxas` (or `ptxas.exe` on Windows)

**Compilation Process:**

```bash
ptxas -arch=sm_90 -O3 input.ptx -o output.cubin
```

**ptxas Options (from `PtxasOptions`):**

Common flags:
- `-arch=sm_XX` - Target architecture
- `-O0/-O1/-O2/-O3` - Optimization level
- `-v` - Verbose output (register usage, memory usage)
- `-lineinfo` - Preserve line information
- `--def-load-cache` - Default load cache modifier
- `--maxrregcount=N` - Maximum register count per thread

**Architecture-Specific Compilation:**

Controlled by `GPUArch` option:
- `sm_90` - Hopper (H100)
- `sm_100a` - Blackwell (B100, B200)
- `sm_101` - Blackwell variant (aliased from sm_110)
- `sm_103` - Blackwell variant
- `sm_120` - Future architecture

**CUBIN Binary Format:**

CUBIN contains:
1. **SASS Machine Code** - Architecture-specific GPU instructions
2. **Metadata:**
   - Register usage per thread
   - Shared memory size
   - Constant memory usage
   - Local memory (stack) size
   - Thread block size requirements
3. **Symbol Table** - Kernel names and entry points
4. **Debug Information** (if `-lineinfo` enabled)
5. **Relocation Information**

**CUBIN Structure:**

```
CUBIN Binary
   ELF Header
   Section Headers
   .text.{kernel_name} - SASS code
   .nv.info - Kernel metadata
   .nv.info.{kernel_name} - Kernel-specific info
   .nv.shared.{kernel_name} - Shared memory layout
   .nv.constant0 - Constant memory
   .symtab - Symbol table
```

**CUBIN Dumping:**

If `KeepCUBIN` option enabled:
- Output path: `{dump_dir}/{function_name}.{arch}.cubin`
- Example: `my_kernel.sm_90.cubin`

**Metadata Example:**

```
Function properties for my_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

.nv.info.my_kernel:
    Attribute: EIATTR_KPARAM_INFO
    Attribute: EIATTR_FRAME_SIZE
        Frame Size = 0
    Attribute: EIATTR_MIN_STACK_SIZE
        Min Stack Size = 0
    Attribute: EIATTR_REGCOUNT
        Registers = 32
    Attribute: EIATTR_MAX_THREADS
        Max Threads = 1024
```

---

### Stage 6: CUBIN � SASS

**Purpose:** Extract and understand the final GPU machine code

**SASS (Shader Assembly):**
- **What it is:** Actual machine code executed by GPU cores
- **Architecture-specific:** Different for each GPU generation (Ampere, Hopper, Blackwell)
- **Binary format:** Embedded in CUBIN
- **Human-readable form:** Via disassembly tools

**SASS is NOT generated separately** - it's contained within the CUBIN binary produced by ptxas.

**Disassembly Tools:**

#### 1. cuobjdump (NVIDIA CUDA Toolkit)

```bash
# Disassemble SASS
cuobjdump -sass my_kernel.sm_90.cubin

# Extract PTX
cuobjdump -ptx my_kernel.sm_90.cubin

# Show ELF sections
cuobjdump -elf my_kernel.sm_90.cubin

# Full disassembly
cuobjdump -all my_kernel.sm_90.cubin
```

#### 2. nvdisasm (NVIDIA CUDA Toolkit)

```bash
# Disassemble with control flow graph
nvdisasm -c my_kernel.sm_90.cubin

# Print function names
nvdisasm -fun my_kernel.sm_90.cubin

# Print line information
nvdisasm -lrm my_kernel.sm_90.cubin

# Print register liveness
nvdisasm -plr my_kernel.sm_90.cubin
```

**Example SASS Output (Hopper sm_90):**

```sass
        Function : my_kernel
    .headerflags    @"EF_CUDA_SM90 EF_CUDA_PTX_SM(EF_CUDA_SM90)"
                                                                               /* 0x001fc400fe2007f6 */
        /*0008*/                   MOV R1, c[0x0][0x28] ;                      /* 0x000000a001017a02 */
        /*0010*/                   S2R R0, SR_TID.X ;                          /* 0x0000000000007919 */
        /*0018*/                   UMOV UR4, 4.0 ;                             /* 0x0000001000047ab2 */
        /*0020*/                   IMAD.WIDE R2, R0, UR4, c[0x0][0x160] ;      /* 0x0580000102007624 */
                                                                               /* 0x001fdc00f40007e4 */
        /*0028*/                   LDG.E.SYS R4, [R2] ;                        /* 0x0000000002047981 */
        /*0030*/                   IMAD.WIDE R2, R0, UR4, c[0x0][0x168] ;      /* 0x05a0000102007624 */
        /*0038*/                   LDG.E.SYS R0, [R2] ;                        /* 0x0000000002007981 */
        /*0040*/                   IMAD.WIDE R2, R0.reuse, UR4, c[0x0][0x170] ;/* 0x05c0000102007624 */
                                                                               /* 0x001fdc00fc0007f6 */
        /*0048*/                   FADD R0, R4, R0 ;                           /* 0x0000000000047221 */
        /*0050*/                   STG.E.SYS [R2], R0 ;                        /* 0x0000000002007986 */
        /*0058*/                   EXIT ;                                      /* 0x000000000000794d */
        /*0060*/                   BRA 0x60 ;                                  /* 0xfffffff000007947 */
```

**SASS Instruction Categories:**

1. **Memory Operations:**
   - `LDG` - Load from global memory
   - `STG` - Store to global memory
   - `LDS/STS` - Shared memory operations
   - `LDC` - Load from constant cache

2. **Arithmetic:**
   - `FADD/FMUL` - Floating-point add/multiply
   - `FFMA` - Fused multiply-add
   - `IMAD` - Integer multiply-add
   - `IMAD.WIDE` - Integer multiply-add with widening

3. **Control Flow:**
   - `BRA` - Branch
   - `EXIT` - Exit kernel
   - `CALL/RET` - Function call/return
   - `BRX` - Indirect branch

4. **Thread Operations:**
   - `S2R` - Special register read (thread ID, warp ID, etc.)
   - `BAR` - Barrier synchronization
   - `SHFL` - Warp shuffle
   - `VOTE` - Warp vote

5. **Special Operations:**
   - `UMOV` - Uniform move
   - `MOV` - Register move
   - `SEL` - Select
   - `PRMT` - Permute bytes

**SASS Encoding:**

Each SASS instruction is encoded in:
- **Hopper (sm_90):** 16 bytes (128 bits) per instruction bundle
- **Ampere (sm_80):** 16 bytes (128 bits) per instruction bundle
- Earlier architectures: 8 bytes (64 bits) per instruction

**Register Types:**
- `R0-R254` - General-purpose registers (32-bit)
- `UR0-UR63` - Uniform registers (constant across warp)
- `P0-P7` - Predicate registers (1-bit)
- `SR_TID.X/Y/Z` - Special registers (thread ID)

**Performance Analysis:**

From SASS, you can determine:
- Instruction count
- Memory access patterns
- Register pressure (# of registers used)
- Warp divergence (branches)
- Memory coalescing efficiency
- Instruction-level parallelism (ILP)

---

### Stage 7: Runtime Execution

**Location:** `python/CuTeDSL/cutlass/base_dsl/jit_executor.py`

**Purpose:** Load CUBIN and execute kernels on GPU

**Process:**

#### 1. CUBIN Extraction from MLIR Module

From `jit_executor.py:73-97`:

```python
def walk_module_and_get_cubin_data(module: ir.Module):
    """Extract CUBIN binary from gpu.binary operations in MLIR module"""
    cubin_data_list = []

    # Walk through module to find gpu.binary operations
    for op in module.body.operations:
        if isinstance(op, gpu.GPUModuleOp):
            for binary_op in op.body.operations:
                if hasattr(binary_op, 'objects'):
                    # Extract CUBIN bytes
                    cubin_data = binary_op.objects[0].value
                    cubin_data_list.append(bytes(cubin_data))

    return cubin_data_list
```

**MLIR Representation:**

```mlir
gpu.module @kernel_module {
  gpu.func @my_kernel(...) kernel { ... }

  gpu.binary @kernel_binary [
    #gpu.object<#nvvm.target<chip = "sm_90">,
                bin = "... CUBIN bytes ...">
  ]
}
```

#### 2. CUDA Module Loading

```python
def load_kernels_from_ir_module(module: ir.Module):
    """Load CUDA modules from CUBIN data"""
    import cuda.bindings.driver as cuda

    cubin_list = walk_module_and_get_cubin_data(module)
    cuda_modules = []

    for cubin_data in cubin_list:
        # Load CUBIN into CUDA runtime
        cuda_module = cuda.cuModuleLoadData(cubin_data)
        cuda_modules.append(cuda_module)

    return cuda_modules
```

**CUDA Driver API Functions:**

1. **Module Management:**
   - `cuModuleLoadData(cubin_bytes)` - Load CUBIN from memory
   - `cuModuleGetFunction(module, name)` - Get kernel function handle
   - `cuModuleUnload(module)` - Unload module

2. **Kernel Execution:**
   - `cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                     shared_mem, stream, args, extras)` - Launch kernel
   - `cuLaunchCooperativeKernel(...)` - Cooperative groups launch
   - `cuLaunchKernelEx(...)` - Extended launch with cluster support

3. **Memory Management:**
   - `cuMemAlloc(size)` - Allocate device memory
   - `cuMemcpyHtoD(dst, src, size)` - Copy host � device
   - `cuMemcpyDtoH(dst, src, size)` - Copy device � host

#### 3. Kernel Launch Configuration

From kernel metadata:

```python
launch_config = {
    'grid': (grid_x, grid_y, grid_z),         # Thread blocks in grid
    'block': (block_x, block_y, block_z),     # Threads per block
    'cluster': (cluster_x, cluster_y, cluster_z),  # Blocks per cluster (Hopper+)
    'shared_mem': smem_bytes,                 # Dynamic shared memory
    'stream': cuda_stream,                    # CUDA stream for async execution
}
```

**Cluster Support (Hopper/Blackwell):**

For architectures sm_90+, clusters enable thread block groups:
```python
# Launch with cluster dimensions
cuLaunchKernelEx({
    'grid': (16, 16, 1),      # 256 thread blocks
    'block': (128, 1, 1),     # 128 threads/block
    'cluster': (4, 4, 1),     # 16 blocks/cluster = 16 clusters total
    'shared_mem': 65536,      # 64KB dynamic shared memory
})
```

#### 4. Execution Flow

```
Python Call
    �
JIT Executor.invoke()
    �
cuLaunchKernel()
    �
CUDA Driver loads CUBIN
    �
GPU Scheduler
    �
SM (Streaming Multiprocessor) executes SASS
    �
Warp Scheduler issues instructions
    �
CUDA Cores execute operations
    �
Results written to memory
    �
Return to Python (synchronization if needed)
```

**Error Handling:**

```python
try:
    result = my_kernel(A, B, C)
except CudaDriverDependencyError:
    # CUDA driver not available
except CompilationError as e:
    # NVVM compilation failed
    print(e.nvvm_error)
    print(e.ir_context)
```

---

## Intermediate Representations

### IR Hierarchy Summary

```
1. Python AST
     Abstract Syntax Tree
     Python language constructs
     Generated by Python's ast module

2. MLIR (CuTe Dialect)
     High-level tensor operations
     Layout transformations
     Memory hierarchy abstractions

3. MLIR (GPU/CUDA Dialects)
     GPU kernel structure
     Thread/block/grid operations
     Memory space annotations

4. MLIR (NVVM Dialect)
     NVIDIA intrinsics
     Architecture-specific operations
     PTX-level constructs

5. PTX (Parallel Thread Execution)
     Virtual ISA
     Text assembly format
     Architecture-independent (within bounds)

6. CUBIN (CUDA Binary)
     Binary container format (ELF)
     Contains SASS machine code
     Metadata and symbol tables

7. SASS (Shader Assembly)
     Actual GPU machine code
     Architecture-specific instructions
     Executed by GPU cores
```

### Example Transformation Trace

**Original Python:**
```python
@cute.jit
def vector_add(A, B, C, N):
    tid = threadIdx.x + blockIdx.x * blockDim.x
    if tid < N:
        C[tid] = A[tid] + B[tid]
```

**MLIR (CuTe Dialect):**
```mlir
gpu.func @vector_add(%A: !cute.tensor, %B: !cute.tensor,
                      %C: !cute.tensor, %N: i32) kernel {
  %tid_x = gpu.thread_id x
  %bid_x = gpu.block_id x
  %bdim_x = gpu.block_dim x
  %tid = arith.addi %tid_x, %bid_x : i32
  %cond = arith.cmpi slt, %tid, %N : i32

  scf.if %cond {
    %a_val = cute.load %A[%tid]
    %b_val = cute.load %B[%tid]
    %sum = arith.addf %a_val, %b_val : f32
    cute.store %sum, %C[%tid]
  }
  gpu.return
}
```

**MLIR (NVVM Dialect):**
```mlir
llvm.func @vector_add(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
                      %arg2: !llvm.ptr, %arg3: i32) {
  %tid_x = nvvm.read.ptx.sreg.tid.x : i32
  %bid_x = nvvm.read.ptx.sreg.ctaid.x : i32
  %bdim_x = nvvm.read.ptx.sreg.ntid.x : i32
  %mul = llvm.mul %bid_x, %bdim_x : i32
  %tid = llvm.add %tid_x, %mul : i32
  %cond = llvm.icmp "slt" %tid, %arg3 : i32

  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %a_ptr = llvm.getelementptr %arg0[%tid] : (!llvm.ptr, i32) -> !llvm.ptr
  %a_val = llvm.load %a_ptr : !llvm.ptr -> f32
  %b_ptr = llvm.getelementptr %arg1[%tid] : (!llvm.ptr, i32) -> !llvm.ptr
  %b_val = llvm.load %b_ptr : !llvm.ptr -> f32
  %sum = llvm.fadd %a_val, %b_val : f32
  %c_ptr = llvm.getelementptr %arg2[%tid] : (!llvm.ptr, i32) -> !llvm.ptr
  llvm.store %sum, %c_ptr : f32, !llvm.ptr
  llvm.br ^bb2
^bb2:
  llvm.return
}
```

**PTX Assembly:**
```ptx
.visible .entry vector_add(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 N
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;

    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra DONE;

    cvt.s64.s32 %rd4, %r5;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f1, [%rd6];

    add.s64 %rd7, %rd2, %rd5;
    ld.global.f32 %f2, [%rd7];

    add.f32 %f3, %f1, %f2;

    add.s64 %rd8, %rd3, %rd5;
    st.global.f32 [%rd8], %f3;

DONE:
    ret;
}
```

**SASS Disassembly (sm_90):**
```sass
        Function : vector_add
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0008*/                   S2R R0, SR_TID.X ;
        /*0010*/                   S2R R2, SR_CTAID.X ;
        /*0018*/                   S2R R3, SR_NTID.X ;
        /*0020*/                   IMAD R0, R2, R3, R0 ;
        /*0028*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x160], PT ;
        /*0030*/              @P0  EXIT ;
        /*0038*/                   SHL R2, R0, 0x2 ;
        /*0040*/                   IMAD.WIDE R2, R0, 4, c[0x0][0x140] ;
        /*0048*/                   LDG.E.SYS R4, [R2] ;
        /*0050*/                   IMAD.WIDE R2, R0, 4, c[0x0][0x148] ;
        /*0058*/                   LDG.E.SYS R5, [R2] ;
        /*0060*/                   FADD R4, R4, R5 ;
        /*0068*/                   IMAD.WIDE R2, R0, 4, c[0x0][0x150] ;
        /*0070*/                   STG.E.SYS [R2], R4 ;
        /*0078*/                   EXIT ;
```

---

## Tools and Libraries

### Python � AST
- **Python `ast` module** - AST parsing and manipulation
- **Python `inspect` module** - Function introspection
- **Custom `DSLPreprocessor`** - AST transformations

### AST � MLIR
- **MLIR Python Bindings** (`cutlass._mlir`)
  - Location: Built from LLVM MLIR project
  - Provides: IR construction, type system, operation creation
- **CuTeDSL Framework**
  - `BaseDSL` - Core DSL infrastructure
  - `CutlassBaseDSL` - CuTe-specific extensions

### MLIR Transformations
- **MLIR PassManager**
  - Orchestrates transformation passes
  - Handles pass dependencies
  - Provides verification
- **MLIR Dialects**
  - Standard dialects: `arith`, `func`, `scf`, `llvm`
  - GPU dialects: `gpu`, `nvvm`, `cuda`
  - Custom dialects: `cute`
- **MLIR ExecutionEngine**
  - JIT compilation infrastructure
  - Runtime code generation

### NVVM � PTX
- **libNVVM** (NVIDIA NVVM Library)
  - Path: `{CUDA_TOOLKIT_PATH}/nvvm/lib/`
  - Files: `libnvvm.so` (Linux), `libnvvm.dll` (Windows)
  - Version: Matches CUDA toolkit version
  - Purpose: Compile NVVM IR to PTX

### PTX � CUBIN/SASS
- **ptxas** (PTX Assembler)
  - Path: `{CUDA_TOOLKIT_PATH}/bin/ptxas`
  - Purpose: Assemble PTX to CUBIN
  - Features: Optimization, register allocation, instruction scheduling

### CUBIN/SASS Inspection
- **cuobjdump** (CUDA Object Dump)
  - Path: `{CUDA_TOOLKIT_PATH}/bin/cuobjdump`
  - Purpose: Disassemble and inspect CUBIN
  - Capabilities: SASS disassembly, PTX extraction, ELF section listing

- **nvdisasm** (NVIDIA Disassembler)
  - Path: `{CUDA_TOOLKIT_PATH}/bin/nvdisasm`
  - Purpose: Advanced SASS disassembly
  - Features: Control flow graphs, register liveness, line info

### Runtime Execution
- **CUDA Driver API** (`cuda.bindings.driver`)
  - Python bindings to CUDA driver
  - Functions: Module loading, kernel launch, memory management
  - Required: NVIDIA GPU driver

- **CUDA Runtime API** (optional)
  - Higher-level API (not used in CuTeDSL's JIT path)
  - Used for runtime compilation in some scenarios

### Supporting Tools
- **nvcc** (NVIDIA CUDA Compiler)
  - Not used in CuTeDSL's MLIR pipeline
  - Used in alternative C++ backend (`cutlass_cppgen`)

- **nvrtc** (NVIDIA Runtime Compilation)
  - Runtime PTX compilation
  - Alternative to JIT path

---

## Code Flow Diagram

### Detailed Execution Flow

```
                                                             
                   Python Source Code                        
                  @cute.jit decorator                        
                       ,                                     
                        
                        �
                                                             
            STAGE 1: AST Preprocessing                       
  File: ast_preprocessor.py                                  
                                                          
   DSLPreprocessor.preprocess()                           
   " Parse Python AST                                     
   " Rewrite control flow                                
   " Transform to IR-friendly form                       
                                                          
                       ,                                     
                        
                        �
                                                             
          STAGE 2: MLIR IR Generation                        
  File: dsl.py, cutlass.py                                   
                                                          
   CutlassBaseDSL._build_ir()                             
   " Create MLIR context                                  
   " Generate gpu.module                                  
   " Trace operations � MLIR ops                          
   " Emit CuTe dialect operations                         
                                                          
                                                             
  Output: MLIR Module (CuTe/GPU/SCF/Arith dialects)          
                       ,                                     
                        
                        �
                                                             
       STAGE 3: MLIR Transformation Passes                   
  File: compiler.py (line 136-194)                           
                                                          
   Compiler.compile()                                     
                                                        
    PassManager.parse(pipeline)                         
    " Pipeline: cute-to-nvvm{options}                   
    " Lower CuTe ops � NVVM intrinsics                 
    " Optimize (opt-level 0-3)                         
    " Apply architecture-specific transforms           
                                                        
   PassManager.run(module.operation)                      
                                                          
                                                             
  Output: MLIR Module (NVVM/LLVM/GPU dialects)               
                       ,                                     
                        
                        �
                                                             
         STAGE 4: PTX Generation                             
  Tool: libNVVM                                              
                                                          
   MLIR NVVM Backend                                      
   " Translate NVVM dialect to PTX                        
   " Apply PTX-level optimizations                        
   " Generate PTX assembly text                           
                                                          
                                                             
  Optional: dump PTX to {name}.{arch}.ptx                    
  Output: PTX Assembly (text)                                
                       ,                                     
                        
                        �
                                                             
       STAGE 5: CUBIN Compilation                            
  Tool: ptxas                                                
                                                          
   ptxas -arch=sm_XX -O{level} input.ptx                 
   " Parse PTX                                            
   " Register allocation                                  
   " Instruction scheduling                               
   " Generate SASS machine code                           
   " Create ELF binary (CUBIN)                            
                                                          
                                                             
  Optional: dump CUBIN to {name}.{arch}.cubin                
  Output: CUBIN Binary (ELF format with SASS)                
                       ,                                     
                        
                        �
                                                             
          STAGE 6: CUBIN Embedded in MLIR                    
                                                          
   MLIR Module Structure:                                 
                                                        
    gpu.module @kernel_module {                         
      gpu.func @my_kernel(...) { ... }                  
      gpu.binary @binary [                              
        #gpu.object<chip="sm_XX",                      
                    bin="<CUBIN bytes>">                
      ]                                                 
    }                                                   
                                                        
                                                          
                                                             
  Output: Complete MLIR Module with embedded CUBIN           
                       ,                                     
                        
                        �
                                                             
         STAGE 7: JIT Execution Engine                       
  File: compiler.py, jit_executor.py                         
                                                          
   Compiler.jit()                                         
   " ExecutionEngine(module, opt_level, shared_libs)      
   " Check CUDA dependencies                              
                                                          
                                                             
                                                          
   walk_module_and_get_cubin_data()                       
   " Extract CUBIN from gpu.binary operations             
                                                          
                                                             
                                                          
   load_kernels_from_ir_module()                          
   " cuModuleLoadData(cubin_bytes)                        
   " cuModuleGetFunction(module, kernel_name)             
                                                          
                       ,                                     
                        
                        �
                                                             
            Kernel Execution on GPU                          
                                                          
   cuLaunchKernel(function, grid, block, ...)             
                                                         
           �                                              
     GPU Scheduler (gigathread engine?)                                       
                                                         
           �                                              
     SM (Streaming Multiprocessor)                        
                                                         
           �                                              
     Warp Scheduler                                       
                                                         
           �                                              
     cuda core or tensor core : Execute SASS Instructions                            
     (Actual machine code on GPU cores)                   
                                                          
                                                             
  Output: Computation results in GPU memory                  
                                                             
```

### Function Call Flow for `cute.compile`

```
cute.compile[options](my_kernel, args)
    
      CompileCallable.__call__()
         CompileCallable._compile()                    [compiler.py:580]
           
             Validate function has _dsl_object
             Process compile options
             Call func._dsl_object._func()
    
      CutlassBaseDSL.__call__()                         [dsl.py]
        
          Stage 1: Build IR
             _build_ir()
                 Create MLIR context
                 Generate gpu.module
                 Trace operations
        
          Stage 2: Compile
             Compiler.compile_and_jit()                [compiler.py:176]
               
                 Compiler.compile()                    [compiler.py:136]
                    PassManager.parse(pipeline)
                    PassManager.run(module)
                    _post_compile_hook (if set)
               
                 Compiler.jit()                        [compiler.py:166]
                     Check CUDA dependencies
                     ExecutionEngine(module)
        
          Stage 3: Extract and Load
              load_kernels_from_ir_module()             [jit_executor.py:73]
                  walk_module_and_get_cubin_data()
                  cuModuleLoadData(cubin)
                  cuModuleGetFunction(module, name)
```

---

## Key Files and Entry Points

### Core Compilation Infrastructure

#### 1. Compiler Entry Point
**File:** `python/CuTeDSL/cutlass/base_dsl/compiler.py`

**Key Functions:**
- `Compiler.compile()` [line 136-194] - Main compilation orchestrator
  - Parses and runs MLIR pass pipeline
  - Handles NVVM compilation errors
  - Invokes post-compile hooks

- `Compiler.jit()` [line 166-174] - JIT execution engine setup
  - Checks CUDA dependencies
  - Creates MLIR ExecutionEngine

- `Compiler.compile_and_jit()` [line 176-193] - Combined flow
  - Calls compile() then jit()

- `CompileCallable._compile()` [line 580-644] - **Your selected function**
  - User-facing compile entry point
  - Processes compile options
  - Delegates to DSL object

**Key Classes:**
- `Compiler` - Manages PassManager and ExecutionEngine
- `CompileOptions` - Encapsulates compilation flags
- `CompileCallable` - Provides `cute.compile` interface
- `CompilationError` - Formatted error reporting

#### 2. DSL Base Infrastructure
**File:** `python/CuTeDSL/cutlass/base_dsl/dsl.py`

**Key Classes:**
- `BaseDSL` - Core DSL implementation
  - IR building
  - Operation tracing
  - Context management

#### 3. CuTe-Specific DSL
**File:** `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py`

**Key Functions:**
- `_get_pipeline()` [line 260-278] - Returns pass pipeline string
  ```python
  "builtin.module(cute-to-nvvm{cubin-format=bin ...})"
  ```

- `preprocess_pipeline()` - Adds CUDA toolkit path and architecture

**Key Classes:**
- `CutlassBaseDSL` - Extends BaseDSL for CuTe
  - Kernel configuration
  - Layout management
  - Tensor operations

#### 4. AST Preprocessing
**File:** `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py`

**Key Classes:**
- `DSLPreprocessor` - AST transformation manager
- `ASTRewriter` - Rewrites Python AST nodes
  - Control flow transformation
  - Function call handling
  - Variable scoping

#### 5. JIT Executor
**File:** `python/CuTeDSL/cutlass/base_dsl/jit_executor.py`

**Key Functions:**
- `walk_module_and_get_cubin_data()` [line 73-97]
  - Extracts CUBIN from MLIR gpu.binary ops
  - Returns list of CUBIN byte arrays

- `load_kernels_from_ir_module()`
  - Loads CUBIN into CUDA runtime
  - Uses cuModuleLoadData() and cuModuleGetFunction()

**Key Classes:**
- `JITExecutor` - Manages kernel execution
  - Argument marshaling
  - Kernel launch
  - Result retrieval

#### 6. NVVM Wrappers
**File:** `python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py`

**Key Functions:**
Wrappers for NVVM intrinsics:
- Thread indexing: `tid_x()`, `tid_y()`, `tid_z()`, `lane_id()`
- Synchronization: `barrier()`, `barrier_arrive()`, `barrier_wait()`
- Warp operations: `shfl_sync()`, `vote_ballot_sync()`
- Memory: `cp_async_commit()`, `cp_async_wait_group()`
- Fence operations: `fence_mbarrier_init()`, `fence_proxy_async()`

### Compilation Options

#### 7. Compile Option Classes
**File:** `python/CuTeDSL/cutlass/base_dsl/compiler.py` [line 240-553]

**Base Classes:**
- `CompileOption` - Base for all options
- `BooleanCompileOption` - True/false flags
- `StringCompileOption` - String-valued options
- `BooleanBasedFileDumpOption` - File output options

**Concrete Options:**
- `OptLevel(0-3)` - Optimization level
- `GPUArch("sm_XX")` - Target architecture
- `PtxasOptions("...")` - Extra ptxas flags
- `EnableAssertions` - Runtime assertions
- `GenerateLineInfo` - Debug line info
- `KeepPTX` - Dump PTX to file
- `KeepCUBIN` - Dump CUBIN to file
- `LinkLibraries("...")` - Link external libraries
- `EnableTVMFFI` - TVM FFI integration

### Supporting Infrastructure

#### 8. Environment Manager
**File:** `python/CuTeDSL/cutlass/base_dsl/env_manager.py`

**Key Class:**
- `EnvironmentVarManager` - Reads environment variables
  - `CUDA_TOOLKIT_PATH`
  - `CUTLASS_DUMP_DIR`
  - `CUTLASS_KEEP_PTX`
  - `CUTLASS_KEEP_CUBIN`
  - `CUTLASS_ARCH`
  - `CUTLASS_ENABLE_ASSERTIONS`
  - `CUTLASS_LINEINFO`

#### 9. MLIR Bindings
**Location:** `python/CuTeDSL/cutlass/_mlir/`

**Modules:**
- `ir` - MLIR IR construction
- `dialects.arith` - Arithmetic ops
- `dialects.func` - Function ops
- `dialects.gpu` - GPU ops
- `dialects.scf` - Control flow ops
- `dialects.cute` - CuTe tensor ops
- `dialects.nvvm` - NVVM intrinsics
- `dialects.cuda` - CUDA runtime ops
- `passmanager` - Pass orchestration
- `execution_engine` - JIT execution

### Alternative Backend (C++ Code Gen)

#### 10. C++ Backend Compiler
**File:** `python/cutlass_cppgen/backend/compiler.py`

**Note:** This is a separate compilation path that generates C++ CUDA code and uses nvcc/nvrtc instead of the MLIR pipeline.

**Key Differences:**
- Generates C++ source code
- Uses nvcc compiler driver
- Different from CuTeDSL's MLIR approach

---

## Configuration Options

### Compilation Flags

#### Optimization Control

**OptLevel(0-3)**
```python
cute.compile[OptLevel(3)](my_kernel)
```
- `0` - No optimization (fastest compile, slowest execution)
- `1` - Light optimization
- `2` - Moderate optimization
- `3` - Aggressive optimization (default)

Affects:
- Register allocation
- Instruction scheduling
- Loop unrolling
- Constant propagation
- Dead code elimination

#### Architecture Selection

**GPUArch("sm_XX")**
```python
cute.compile[GPUArch("sm_90")](my_kernel)
```

Supported architectures:
- `sm_80` - Ampere (A100)
- `sm_86` - Ampere (RTX 30xx)
- `sm_89` - Ada Lovelace (RTX 40xx, L40)
- `sm_90` - Hopper (H100)
- `sm_90a` - Hopper with FP8 Tensor Core support
- `sm_100a` - Blackwell (B100, B200)
- `sm_101` - Blackwell variant
- `sm_103` - Blackwell variant
- `sm_120` - Future architecture

**Note:** `sm_110` is automatically converted to `sm_101`

#### PTX Assembler Options

**PtxasOptions("...")**
```python
cute.compile[PtxasOptions("-v --maxrregcount=128")](my_kernel)
```

Common ptxas flags:
- `-v` - Verbose (show register/memory usage)
- `--maxrregcount=N` - Limit registers per thread
- `--def-load-cache` - Default load cache modifier
- `--gpu-name=sm_XX` - Override GPU name
- `--opt-level=N` - PTX optimization level

### Debugging Options

#### Debug Information

**GenerateLineInfo**
```python
cute.compile[GenerateLineInfo](my_kernel)
```
- Preserves source line information
- Enables profiling with source correlation
- Increases binary size
- Adds `-lineinfo` flag to ptxas

#### Runtime Assertions

**EnableAssertions**
```python
cute.compile[EnableAssertions](my_kernel)
```
- Enables runtime assertion checks
- Validates tensor bounds, thread indices, etc.
- Adds runtime overhead
- Useful for development/debugging

### Artifact Dumping

#### PTX Dumping

**KeepPTX**
```python
cute.compile[KeepPTX](my_kernel)
```
- Saves generated PTX to file
- Path: `{dump_dir}/{function_name}.{arch}.ptx`
- Useful for:
  - Inspecting virtual ISA
  - Debugging compilation issues
  - Understanding optimization effects

#### CUBIN Dumping

**KeepCUBIN**
```python
cute.compile[KeepCUBIN](my_kernel)
```
- Saves compiled CUBIN to file
- Path: `{dump_dir}/{function_name}.{arch}.cubin`
- Can be disassembled with cuobjdump/nvdisasm
- Useful for:
  - SASS inspection
  - Performance analysis
  - Binary distribution

#### Dump Directory

**Environment Variable:**
```bash
export CUTLASS_DUMP_DIR=/path/to/dump
```

Default: Current working directory

### Advanced Options

#### Library Linking

**LinkLibraries("...")**
```python
cute.compile[LinkLibraries("libcudadevrt.a")](my_kernel)
```
- Link external CUDA libraries
- Supports static libraries (.a)
- Useful for:
  - Device-side runtime functions
  - Custom CUDA libraries

#### TVM FFI Integration

**EnableTVMFFI**
```python
cute.compile[EnableTVMFFI](my_kernel)
```
- Enables Apache TVM FFI support
- Requires: `pip install apache-tvm-ffi`
- Allows TVM to call CuTe kernels

### Environment Variables

**Complete List:**

```bash
# Required
export CUDA_TOOLKIT_PATH=/usr/local/cuda

# Optional debugging
export CUTLASS_DUMP_DIR=/tmp/cutlass_dump
export CUTLASS_KEEP_PTX=1
export CUTLASS_KEEP_CUBIN=1
export CUTLASS_LINEINFO=1
export CUTLASS_ENABLE_ASSERTIONS=1

# Architecture override
export CUTLASS_ARCH=sm_90

# TVM integration
export CUTLASS_ENABLE_TVM_FFI=1
```

### Option Composition

**Multiple Options:**
```python
cute.compile[
    OptLevel(3),
    GPUArch("sm_90"),
    KeepPTX,
    KeepCUBIN,
    GenerateLineInfo,
    PtxasOptions("-v"),
](my_kernel, args)
```

**String-Based Options (alternative syntax):**
```python
cute.compile(my_kernel, args, options="--opt-level=3 --gpu-arch=sm_90 --keep-ptx --keep-cubin")
```

### Default Values

If not specified:
- `OptLevel`: 3 (aggressive optimization)
- `GPUArch`: Read from `CUTLASS_ARCH` or auto-detect
- `PtxasOptions`: "" (none)
- `EnableAssertions`: False
- `GenerateLineInfo`: False
- `KeepPTX`: False
- `KeepCUBIN`: False

---

## Debugging and Inspection

### Inspecting Intermediate Representations

#### 1. View MLIR IR

**Enable MLIR dumping:**
```python
import os
os.environ['MLIR_ENABLE_DUMP'] = '1'

@cute.jit
def my_kernel(...):
    ...

# MLIR will be printed to stderr during compilation
```

**Access IR programmatically:**
```python
# After compilation, access the module
executor = cute.compile[options](my_kernel, args)
module = executor._module  # Access MLIR module
print(module)
```

#### 2. Dump PTX

**Method 1: Compile option**
```python
cute.compile[KeepPTX](my_kernel, args)
# PTX saved to: {dump_dir}/my_kernel.sm_90.ptx
```

**Method 2: Environment variable**
```bash
export CUTLASS_KEEP_PTX=1
export CUTLASS_DUMP_DIR=/tmp/ptx_dump
```

**Inspect PTX:**
```bash
cat /tmp/ptx_dump/my_kernel.sm_90.ptx
```

#### 3. Dump and Inspect CUBIN

**Dump CUBIN:**
```python
cute.compile[KeepCUBIN](my_kernel, args)
# CUBIN saved to: {dump_dir}/my_kernel.sm_90.cubin
```

**Disassemble SASS with cuobjdump:**
```bash
# Show SASS assembly
cuobjdump -sass my_kernel.sm_90.cubin

# Show all information
cuobjdump -all my_kernel.sm_90.cubin

# Extract PTX from CUBIN
cuobjdump -ptx my_kernel.sm_90.cubin

# Show ELF sections
cuobjdump -elf my_kernel.sm_90.cubin
```

**Disassemble with nvdisasm:**
```bash
# Basic disassembly
nvdisasm my_kernel.sm_90.cubin

# With control flow graph
nvdisasm -c my_kernel.sm_90.cubin

# Show register liveness
nvdisasm -plr my_kernel.sm_90.cubin

# Show line information (if compiled with GenerateLineInfo)
nvdisasm -lrm my_kernel.sm_90.cubin
```

### Performance Analysis

#### 1. Register and Memory Usage

**From ptxas output:**
```python
cute.compile[PtxasOptions("-v")](my_kernel, args)
```

Output shows:
```
ptxas info    : Compiling entry function 'my_kernel' for 'sm_90'
ptxas info    : Function properties for my_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, 65536 bytes smem, 344 bytes cmem[0]
```

Metrics:
- **Registers:** Per-thread register usage (affects occupancy)
- **Spill stores/loads:** Register pressure (slow!)
- **Shared memory:** Per-block shared memory usage
- **Constant memory:** Constant cache usage

#### 2. Occupancy Calculation

**Formula:**
```
Occupancy = (Blocks per SM) / (Max Blocks per SM)
```

**Limiting factors:**
- Registers per thread
- Shared memory per block
- Threads per block
- Max threads per SM

**Tools:**
- NVIDIA Occupancy Calculator
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`

**Example:**
```
SM: Hopper H100
Max threads/SM: 2048
Max registers/SM: 65536
Max shared mem/SM: 228 KB

Kernel config:
- 128 threads/block
- 32 registers/thread
- 64 KB shared mem/block

Blocks/SM limited by registers: 65536 / (128 * 32) = 16
Blocks/SM limited by shared mem: 228 KB / 64 KB = 3
Blocks/SM limited by threads: 2048 / 128 = 16

Actual blocks/SM: min(16, 3, 16) = 3
Occupancy: 3 / 16 = 18.75%
```

#### 3. Profiling with NVIDIA Nsight

**Nsight Compute (kernel profiling):**
```bash
ncu --set full -o profile python my_script.py
```

Metrics:
- SASS instruction mix
- Memory throughput
- Warp divergence
- Occupancy
- Stall reasons

**Nsight Systems (timeline profiling):**
```bash
nsys profile -o timeline python my_script.py
```

Shows:
- Kernel launches
- CPU-GPU synchronization
- Memory transfers
- CUDA API calls

### Debugging Compilation Errors

#### 1. NVVM Errors

**Error format:**
```
NVVM Compilation Error:
----------------------

�  Current Settings:
- CUDA Toolkit Path: /usr/local/cuda
- Target Architecture: sm_90

IR Context (truncated):
  %0 = nvvm.read_ptx_sreg_tid_x : i32
  ...

=� Possible Solutions:
1. Check if CUDA_TOOLKIT_PATH is set correctly
2. Verify target architecture (sm_90) is supported by your CUDA toolkit
3. Make sure CUDA toolkit version matches the target architecture requirements
```

**Common causes:**
- Unsupported architecture for CUDA version
- Missing CUDA toolkit
- Incorrect `CUDA_TOOLKIT_PATH`
- Invalid NVVM operations

**Solutions:**
- Update CUDA toolkit
- Set correct architecture: `GPUArch("sm_80")`
- Verify: `export CUDA_TOOLKIT_PATH=/usr/local/cuda`

#### 2. PTX Errors

**Error from ptxas:**
```
ptxas error   : Entry function '_Z9my_kernelPfS_S_' uses too much shared data (0x20000 bytes, 0x18000 max)
```

**Common causes:**
- Exceeded shared memory limit
- Too many registers
- Invalid PTX syntax (internal error)

**Solutions:**
- Reduce shared memory usage
- Lower `--maxrregcount`
- Reduce block size

#### 3. Runtime Errors

**CUDA Driver errors:**
```python
try:
    result = my_kernel(A, B, C)
except CudaDriverDependencyError as e:
    print("CUDA driver not available or GPU not found")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

**Common errors:**
- `CUDA_ERROR_INVALID_VALUE` - Invalid kernel arguments
- `CUDA_ERROR_OUT_OF_MEMORY` - Insufficient GPU memory
- `CUDA_ERROR_LAUNCH_FAILED` - Kernel launch failed (check assertions)

### Advanced Debugging Techniques

#### 1. Enable Verbose Logging

```python
import logging
from cutlass.base_dsl.utils.logger import log

log().setLevel(logging.DEBUG)
```

#### 2. Custom Post-Compile Hook

```python
from cutlass.base_dsl.compiler import PostCompileHookContext

def my_hook(module):
    print("=" * 80)
    print("MLIR Module after compilation:")
    print(module)
    print("=" * 80)

with PostCompileHookContext(compiler, my_hook):
    cute.compile[options](my_kernel, args)
```

#### 3. SASS Instruction Analysis

**Count instruction types:**
```bash
nvdisasm my_kernel.cubin | grep -oP '^\s+/\*[0-9a-f]+\*/\s+\K[A-Z]+' | sort | uniq -c
```

Output:
```
     45 LDG
     38 STG
     12 FADD
     18 FMUL
     24 FFMA
      8 BAR
     ...
```

**Identify memory access patterns:**
```bash
nvdisasm my_kernel.cubin | grep -E 'LDG|STG|LDS|STS'
```

#### 4. Compare Different Optimization Levels

```bash
# Compile with different opt levels
cute.compile[OptLevel(0), KeepCUBIN](kernel, args)  # kernel.sm_90.cubin
cute.compile[OptLevel(3), KeepCUBIN](kernel, args)  # kernel.sm_90.cubin

# Compare SASS
diff <(nvdisasm kernel_O0.cubin) <(nvdisasm kernel_O3.cubin)
```

---

## Summary

The CuTeDSL compilation pipeline transforms Python code into GPU machine code through seven distinct stages:

1. **Python � AST:** Parse and optionally rewrite Python syntax
2. **AST � MLIR (CuTe):** Generate high-level tensor IR
3. **CuTe � NVVM:** Lower to GPU intrinsics via `cute-to-nvvm` pass
4. **NVVM � PTX:** Generate virtual ISA assembly (libNVVM)
5. **PTX � CUBIN:** Assemble to binary with SASS (ptxas)
6. **SASS:** Final machine code (embedded in CUBIN)
7. **Runtime:** Load and execute on GPU (CUDA Driver API)

**Key transformation point:** The `Compiler.compile()` method at `compiler.py:136-194` orchestrates the MLIR pass pipeline, applying the `cute-to-nvvm` transformation and subsequent optimizations.

**Tools used:**
- MLIR for flexible IR transformations
- libNVVM for PTX generation
- ptxas for SASS compilation
- cuobjdump/nvdisasm for inspection
- CUDA Driver API for execution

**Configuration:** Extensive options via `CompileOptions` and environment variables control optimization level, target architecture, debugging features, and artifact dumping.

This multi-stage pipeline enables high-level Python programming with CuTe abstractions while generating efficient GPU machine code competitive with hand-written CUDA.

---

## References

### Documentation
- MLIR Documentation: https://mlir.llvm.org/
- CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVVM IR: https://docs.nvidia.com/cuda/nvvm-ir-spec/

### Tools
- cuobjdump: `{CUDA_TOOLKIT}/bin/cuobjdump`
- nvdisasm: `{CUDA_TOOLKIT}/bin/nvdisasm`
- ptxas: `{CUDA_TOOLKIT}/bin/ptxas`
- Nsight Compute: https://developer.nvidia.com/nsight-compute
- Nsight Systems: https://developer.nvidia.com/nsight-systems

### CuTeDSL Source Files
- Compiler: `python/CuTeDSL/cutlass/base_dsl/compiler.py`
- DSL Base: `python/CuTeDSL/cutlass/base_dsl/dsl.py`
- CuTe DSL: `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py`
- AST Preprocessor: `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py`
- JIT Executor: `python/CuTeDSL/cutlass/base_dsl/jit_executor.py`
- NVVM Wrappers: `python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py`
