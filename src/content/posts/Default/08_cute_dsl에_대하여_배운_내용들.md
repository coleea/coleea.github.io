---
title: cute dsl에 대하여 배운 내용들
published: 0001-01-01
description: ""
tags: []
category: Default
draft: true
---

CuTe DSL은 파이썬 언어 기반의 dsl이다. 이 DSL은 최종적으로 SASS ISA로 변환되어 GPU에서 실행된다.

- cute dsl 컴파일 흐름

이 변환 작업은 CuTe의 JIT 컴파일러가 수행한다.
- 파이썬에서 데코레이터로 jit컴파일러 기호를 삽입한다
- cute dsl의 컴파일러가 실행된다. cutlass 리포지토리에서 컴파일러 코드의 경로는 다음과 같다 : `python/cutlass\base_dsl\compiler.py`

이 파일에는 다음과 같이 적혀있다.
This module provides a class that compiles generated IR using MLIR's `PassManager` and executes it using MLIR's `ExecutionEngine`.

- 즉 MLIR's PassManager은 인자로 넘겨받는 코드를 IR로 변경한다
```
pm = self.passmanager.PassManager.parse(pipeline)
pm.enable_verifier(enable_verifier)
pm.run(module.operation)
```
PassManager.parse
pass_messenger.run(module.operation)

(qustion : which IR exactly ? MLIR ?)
question : where NVVM(nvidia virtual machine) is used ?

- 그리고 MLIR의 ExecutionEngine을 통하여 이 코드를 실행한다. 이 엔진은 just-in-time 엔진이다. 
즉 파이썬 데코레이터인 `@cute.jit`는 결국 MLIR의 ExecutionEngine을 사용한다는 의미이다

qustion : this ExecutionEngine use nvcc compiler for creating SASS ISA ?

---
