---
title: WebGPU기반의 AI 추론시 Nvidia GPU까지의 실행흐름 정리
published: 0001-01-01
description: ""
tags: []
category: Default
draft: false
---

- TODO
- sm 유닛 사진을 블랙웰 사진으로 교체할 것
- 강의 영상에서 512 ROW에 대한 설명이 잘못 적혀있는 부분을 수정할 것
- WebGPU의 WGSL 코드를 컴파일하면 어떤 코드가 나오는지 보여줄것

---

# 크롬 웹브라우저에서 WebGPU API를 호출시 발생하는 실행흐름 정리

## 웹 브라우저 파이프라인 

블링크 렌더러 프로세스의 메인 스레드에서는 자바스크립트 엔진이 WebGPU API를 호출한다

WebGPU는 2가지 페이즈로 구성된다. 

첫번째 : 파이프라인 생성 페이즈 : `createComputePipeline` 함수를 호출할 때, the IR (WGSL compiled to
         SPIR-V/DXIL/MSL) is passed to the driver. The driver compiles this into GPU machine code (SASS) at this stage.

Queue Submit (Execution Phase): When you call device.queue.submit(), you are passing a Command Buffer. This buffer contains
         "draw" or "dispatch" commands that refer to the already-compiled pipeline. It does not transfer the shader code (IR) again.

`CreateCommandEncoder`를 이용해 커맨드 인코더를 생성하고, 데이터 버퍼를 준비합니다.

WebGPU API인 `device.queue.submit()`은 블링크 내부의 WebGPU 구현체인 "Dawn"에게 제어권을 넘긴다

"Dawn"에 구현된 Submit 함수는 mojo IPC를 통하여 직렬화된 데이터를 GPU 프로세스로 전송한다. 직렬화된 데이터에는 커맨드 인코더와 데이터 버퍼에 대한 정보가 포함되어 있다
컴퓨트 워크로드의 경우 이 과정에서 DOM 렌더링 파이프라인과 CC 스레드 워크플로우를 우회한다
(단, WebGPU로 캔버스 렌더링을 하는 경우엔 최종적으로 Compositor와 조율한다 : Compositor와 조율한다는게 무슨 뜻인가 ? 그건 바로 : GPU에서 렌더링된 이미지를 모니터에 출력하기 위해 Compositor와 협력한다는 뜻)
(단, WebGPU로 캔버스 렌더링을 하는 경우, Dawn이 생성한 텍스처(Swap Chain)는 브라우저의 Compositor(Viz 프로세스)에게 전달되어 페이지의 다른 요소들과 합성된 후 화면에 출력됩니다.)

GPU 프로세스에서는 "Dawn"에게 수신받은 직렬화된 데이터를 역직렬화하여 WebGPU 커맨드 버퍼를 재구성한다 
GPU 프로세스에서 Dawn 내부에 구현된 Tint 컴파일러를 호출하여 WGSL(WebGPU Shading Language) 코드를 운영체제별 IR 코드로 컴파일한다

- Windows의 경우: WGSL → HLSL → DXIL (DirectX 12 Intermediate Language)
- Linux의 경우: WGSL → SPIR-V (Vulkan Intermediate Representation)
- macOS의 경우: WGSL → MSL → AIR (Apple Intermediate Representation for Metal)

이후 운영체제 레벨의 그래픽 API(Vulkan, Metal, D3D12)를 호출하여 GPU 명령을 제출합니다.
(변환된 IR 코드는 '파이프라인 상태 객체(PSO)'를 생성할 때 드라이버에 전달되며, 실제 그리기/연산 명령을 담은 '커맨드 버퍼'는 이 PSO를 참조하여 실행된다)   

이 추상화 계층은 OS 커널에서 관리하는 GPU 드라이버에게 커맨드 버퍼를 전달한다

GPU 드라이버는 전달받은 커맨드 버퍼를 해석하여 Nvidia RTX GPU용 `커널 드라이버 API`를 호출하고

커널 드라이버는 libcuda 내부에 있는 JIT 컴파일러를 사용하여 GPU에 최적화된 **SASS (Shader Assembly)** 머신 코드를 생성합니다.              

같은 WebGPU 코드라고 해도 GPU 아키텍처에 따라서 최종 생성되는 SASS 코드가 달라진다
(PTX 명령어인 wmma(Warp-Level Matrix Multiply-Accumulate) 등도 이 단계에서 SASS로 변환됨)

이 SASS 코드는 Nvidia GPU의 물리적 하드웨어에서 실행된다
(질문 GPU 내부에서 이 SASS 코드의 실행흐름을 관리하는 유닛은 무엇인가 ? )

CPU는 GPU 하드웨어에 커널 런치 커맨드를 전송한다 대략 아래와 같은 커맨드가 전송된다

```
launch(thread_block_dimension, convolve)
```

![](/images/gpu_nvidia/kernel_launch_command.png)
이미지 출처 : https://youtu.be/qQTDF0CBoxE?si=AHORGY6jDt2cWNhn&t=2974

## Nvidia GPU 물리적 실행 파이프라인

![](/images/gpu_nvidia/turing-architecture.jpg)
튜링 아키텍처 기반 GPU의 실물사진. 출처 : [Nvidia Technical Blog](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)

Nvidia GPU 내부에서 SASS 명령어가 실행되는 과정을 단계별로 살펴보자

### 1. GPU System Processor (GSP)

![](/images/gpu_nvidia/gsp.avif)
출처 : https://riscv.org/blog/how-nvidia-shipped-one-billion-risc-v-cores-in-2024/

 - 위에서 호스트(CPU)가 GPU에게 커널 런치 커맨드를 전송했다. 이 커맨드를 받아들이는 유닛이 GSP이다.
 - GSP는 GPU 내부의 "관리자"이자 "게이트키퍼" 역할을 수행하는 RISC-V 기반의 보조 프로세서(Co-processor)이다. 구버전에는 존재하지 않은 하드웨어지만 2018년 출시된 튜링  아키텍처 (RTX 2000 시리즈) 부터 도입되었다.
 - "게이트키퍼"라는 건 GPU외부에 있는 데이터가 GSP를 통해 GPU 내부로 진입한다는 뜻이다
 - 왜 "게이트키퍼" 인지는 위 사진에서 확인할 수 있을 것이다. 커널 드라이버는 PCI Express 레인을 통하여 전기적 신호를 GPU에게 보내는데 이 신호를 처리하는 유닛이 GSP인 것이다.
 - 또한 기존에 CPU의 커널 모드 드라이버(RM)가 수행하던 GPU 관리 작업을 GPU 내부에서 수행하기 위해 도입되었다. 
 - 즉 전력 관리, 열 관리, 클럭(Hz) 실시간 제어 등 하드웨어의 저수준 제어를 담당하는데, 터미널에서 `nvidia-smi` 를 입력하면 표시되는 GPU 세부 정보는 이 GSP 유닛이 응답하는 정보들이다.
 - 이렇게 GPU 내부에서 관리 작업을 처리하여 얻을 수 있는 이점은 CPU-GPU 간의 통신 지연(Latency)을 최소화할 수 있다는 것이다.
 - 이런 메타정보 관리 외에도 CPU가 보낸 명령 스트림을 받아 커맨드 버퍼 파싱을 담당한다. 이렇게 파싱된 데이터는 하드웨어 스케줄러(GigaThread Engine)에 전달된다.

### 2. GigaThread Engine

![](/images/gpu_nvidia/h100-gigathread.png)
GH100의 구조. 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

- 이름이 의미하는 대로 스레드 엔진이며 일종의 스레드 스케줄러이다
- 위 이미지에서 최상단에 위치해 있는데 `GigaThread Engine with MIG Control` 이라고 적혀있다
- MIG는 Multi-Instance GPU의 약자이다. which enables the physical partitioning of a single GPU into multiple, isolated, and securely independent instances
- 기가스레드 엔진은 GSP로부터 커널 런치(kernel launch) 커맨드를 수신받는다.
- 기가스레드 엔진은 전달받은 커널을 분석한다
- 커널 분석이 완료되면 워크 디스패치 작업을 수행한다
- 워크 디스패치는 스레드 블록(다른말로 Work, Workgoup이라고도 부른다) 또는 스레드 블록 클러스터를 생성하는 작업을 의미한다.
- 기가스레드 엔진은 커널 코드에 적혀있는 스레드블록 관련 정보를 읽어들여 스레드 블록을 생성한다.
- 예를 들어 스레드 블록이 <16,8,1>이라고 적혀있으면 이것을 곱한 16*8*1 = 128개의 스레드로 처리되는 스레드 블록에 대한 데이터를 생성한다
- 스레드 블록의 최대 차원은 1024이며 차원이 1025이상인 값은 설정 불가능하다
- 스레드 블록 또는 스레드 블록 클러스터 생성이 완료되었으면 스케줄러가 특정 SM에 스레드 블록을 (또는 스레드 블록 클러스터를) 전달한다 
- 구체적인 흐름도는 아래와 같다

```
Gigathread Engine
    ↓
Sends control packet/signal to SM via interconnect
    ↓
SM's Block Scheduler receives signal
    ↓
Signal contains:
- Thread block ID (blockIdx.x, y, z)
- Thread block dimensions (blockDim.x, y, z)
- Grid dimensions (gridDim.x, y, z)
- Kernel program counter (where to start execution)
- Resource requirements (registers, shared memory per thread/block)
```

```
┌──────────────────────────┐
│   Gigathread Engine      │
│  "Assign TB 42 to SM 5"  │
└───────────┬──────────────┘
            │ Control Packet
            ↓
┌──────────────────────────────────┐
│         SM 5                     │
│  ┌────────────────────────────┐ │
│  │ Block Scheduler (receives) │ │
│  └──────────┬─────────────────┘ │
│             ↓                    │
│  ┌──────────────────────────┐  │
│  │ Resource Allocator        │  │
│  │ - Check availability      │  │
│  │ - Allocate registers      │  │
│  │ - Allocate shared memory  │  │
│  └──────────┬───────────────┘  │
│             ↓                    │
│  ┌──────────────────────────┐  │
│  │ Warp Generator            │  │
│  │ - Create 8 warps          │  │
│  │ - Initialize warp state   │  │
│  └──────────┬───────────────┘  │
│             ↓                    │
│  ┌──────────────────────────┐  │
│  │ Warp Pool                 │  │
│  │ [W0,W1,W2,...,W7]         │  │
│  └──────────────────────────┘  │
└──────────────────────────────────┘
```

- 이 유닛은 3가지 기능을 수행한다 첫번째가 워크 디스패치이고 두번째가 스레드 스케줄링이다. 세번째는 컨텍스트 스위칭이다.
- 주의 : 기가스레드 엔진이 프로그램 명령어를 스레드에 할당하는 단위는 스레드 블록, 또는 스레드 블록 클러스터 단위이다. "절대로" Warp단위로 할당하지 않는다

"here is an instruction stream, a program binary. please run all of these thread blocks of it."
"i have to run 8,000 thread blocks. I've got 4 cores, let's get to work"
출처 : https://youtu.be/qQTDF0CBoxE?si=H0wXGXyt4y-ErSeT&t=2985

#### 2.1 컨텍스트 스위칭

- 기가스레드 엔진은 컨텍스트 스위칭도 담당하는데, 이는 하드웨어 자원 사용률을 극대화하려는 목적으로 사용된다.
  즉 Warp에서 메모리 억세스를 수행할 때 이 메모리 레이턴시(Memory Latency) 기간동안 다른 대기중인 작업을 Warp에 할당하여 연산 유닛의 유휴 시간을 최소화한다. 
 
### 3. GPU Processing Cluster (GPC)

![](/images/gpu_nvidia/h100_gpc.png)
GH100의 GPC 구조. 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

- 여러 Streaming Multiprocessor (SM)들을 묶는 상위 물리적 단위이다. 즉 SM보다 더 상위 레벨의 병렬 처리를 제공한다. 
- 이 개념은 구형 아키텍처를 배운 분들한테는 생소할 수 있는데 2022년 공개된 Hopper 아키텍처에 처음 도입된 개념이기 때문이다. 
- 왜 GPU Processing Cluster (GPC)가 도입되었는가 ? 아래 그림을 보자

![](/images/gpu_nvidia/thread_block_cluster_1.jpg)
(출처 : [Nvidia’s H100 GPU Seminar by Jack Choquette](https://youtu.be/MC223HlPdK0?si=qsb1gTV1QM_3Qia-&t=580))

- 이것은 GPC가 도입되기 전의 아키텍처이다. SM이 다른 SM의 데이터를 참조하려면 반드시 L2 캐시를 거치거나 L2 캐시가 미스되면 글로벌 메모리에 억세스해야 하는 구조로 되어있다.
- 글로벌 메모리 억세스는 매우 긴 레이턴시를 소비하므로 필연적인 속도 저하를 가져온다. 그래서 고안해 낸 개념이 글로벌 메모리에 억세스하는 대신 SM과 SM간의 직접적인 데이터 전송을 하고 싶었던 것이다
- 그렇게 해서 고안된 새로운 아키텍처가 아래와 같다

![](/images/gpu_nvidia/thread_block_cluster_2.jpg)
(출처 : [Nvidia’s H100 GPU Seminar by Jack Choquette](https://youtu.be/MC223HlPdK0?si=qsb1gTV1QM_3Qia-&t=580))

- 바로 SM과 글로벌 메모리 중간에 GPC라는 레이어를 추가하여 이 GPC에서 SM과 SM간의 데이터 전송을 담당하는 것이다
- GPC 내부에는 SM 하드웨어간의 전용 회로가 존재하여 고속의 데이터 공유가 가능하다. 
- SM간의 데이터 전송은 SM 내부에 있는 텐서 메모리 엑셀레이터(Tensor Memory Accelerator, TMA)라 불리는 유닛에 의해 수행된다. 즉 SM과 SM간의 데이터 전송은 실제로 TMA와 TMA 간의 데이터 전송이다.
- 그리고 이 TMA와 TMA간의 데이터 전송을 관장하는 유닛이 GPC이다. 
- 데이터 전송 외에도 GPC는 클러스터 레벨의 Barrier 기능을 제공한다. 즉 여러 SM들이 동시에 프로그램의 특정 지점에 도달했는지 확인하는 기능을 제공하는데 이건 Hopper 아키텍처 이전에는 불가능했던 기능이다.
- 즉 GPC는 SM에서 처리하는 스레드 블록(Thread Block)보다 더 상위의 개념인 스레드 블록 클러스터(Thread Block Cluster) 단위의 병렬처리를 수행하기 위하여 만들어진 하드웨어가 GPU Processing Cluster (GPC)이다.
- 또한 스레드 블록과 마찬가지로 스레드 블록 클러스터 또한 프로그래밍 레벨에서 컨트롤이 가능하다. 
- 예를들어 쿠다 커널의 `__cluster_dims__(X,Y,Z)` 어트리뷰트를 사용하여 스레드 블록 클러스터의 크기를 지정할 수 있다. 여기서 설정가능한 차원의 최대값은 8이다. 다시 말해 GPC에서 동시에 병렬처리가 가능한 SM의 갯수는 최대 8개라는 의미이다.
- 예제 코드로 확인해 보자

```cuda 
__cluster_dims__(4,2,1) // 총 8개의 스레드 블록을 클러스터로 설정한다
__global__ void helloCluster() {
	cooperative_groups::cluster_group cluster = this_cluster();

	// 8개 스레드 블록의 모든 스레드가 작업을 처리할 때 까지 대기한다
	// 이 동기화 함수는 GPC에서 제공하는 바리어(barrier) 전용 하드웨어를 사용한다
	cluster.sync(); 

	printf("hello from cluster element of %d\n", cluster.cluster_rank());
}
```

예제코드 출처 : https://youtu.be/MC223HlPdK0?si=2fdNeZcea4hkYV5G&t=806

한가지 이상한 점은, [쿠다코어 가이드](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)에 의하면 클러스터로 설정 가능한 스레드 블록의 최대 갯수는 8이라고 적혀있는데 스탠포드 세미나에 의하면 16개의 스레드 블록을 하나의 클러스터로 설정할 수 있다는 것이다

> The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA.

출처 : https://docs.nvidia.com/cuda/cuda-c-programming-guide/

반면 Nvidia의 Senior Distinguished Engineer인 Jack Choquette는 16개의 스레드를 클러스터로 설정할 수 있다고 말한다
![](/images/gpu_nvidia/thread_block_cluster_max_size.jpg)

- 위의 2가지 주장 중에서 어느것이 사실인지 확인이 필요해 보인다

- 다시말해 GPC는 스레드 블록 클러스터(Thread Block Cluster)에 대한 스케줄러로서, 클러스터 내의 Thread Block들을 내부 SM들에 분배하는 역할을 담당한다.

- 여기까지 들으면 헷갈릴 수 있으므로 GPU의 병렬처리 계층을 다시 설명하겠다 
- 태초에 스레드가 있다. 이 스레드는 프로그램을 실행하는 최소 단위이다

![](/images/gpu_nvidia/thread.png)
(이미지 출처 : [유튜브 채널 Branch Education](https://www.youtube.com/watch?v=h9Z4oGN89MU))

- 하지만 GPU는 병렬 처리에 특화된 하드웨어이므로 여러 스레드를 하나의 단위로 묶어서 동시에 실행시킬 필요가 있다. 여기서 묶인 단위를 "워프(Warp)"라고 부른다

![](/images/gpu_nvidia/warp.png)
(이미지 출처 : [유튜브 채널 Branch Education](https://www.youtube.com/watch?v=h9Z4oGN89MU))

- 그리고 여러개의 워프를 또 하나의 단위로 묶어서 병렬 처리의 단위로 컨트롤할 필요성이 생겼다. 이를 "스레드 블록(Thread Block)"이라고 부른다

![](/images/gpu_nvidia/thread_block.png)
(이미지 출처 : [유튜브 채널 Branch Education](https://www.youtube.com/watch?v=h9Z4oGN89MU))

- 그리고 여러개의 스레드 블록을 또 하나의 단위로 묶어서 병렬 처리의 단위로 컨트롤할 필요성이 생겼다. 이를 "스레드 블록 클러스터(Thread Block Cluster)"라고 부른다

![](/images/gpu_nvidia/thread_block_cluster.png)
(이미지 출처 : https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

- 이 여러개의 스레드 블록 클러스터를 모아서 그리드(Grid)라고 부른다. 

GPU Processing Cluster (GPC)에 대한 상세 내용을 알고 싶으으면 [Nvidia Blog의 소개글](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) 또는 [CUDA 가이드](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 를 참조하라

### 4. Streaming Multiprocessor (SM)

![](/images/gpu_nvidia/h100-sm.png)
GH100 내부의 SM 구조. 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

![](/images/gpu_nvidia/v100_gpu_sm.png)
Volta 100의 SM 구조 : 출처 : https://youtu.be/qQTDF0CBoxE?si=nzQGQz42Jv5xJG-x&t=3640

- SM은 다른 말로 코어(Core)라고도 불린다
- SM은 GPC로부터 스레드 블록(Thread Block)을 전달받는다

- GPC로부터 스레드 블록을 전달받으면 SM은 이 스레드 블록의 스레드 갯수를 32로 나눈다
- 32는 워프(Warp)가 실행되는 스레드의 단위이므로 스레드 블록의 스레드 갯수를 32로 나눈 값은 워프의 갯수이다.
- 그리고 이렇게 생성된 워프들을 워프 풀(Warp Pool)에 보관한다. 
- 이 스레드 풀은 위의 이미지에는 표시되지 않았지만 SM 내부에 엄연히 존재하는 하드웨어 유닛이다
- 이 워프 풀을 참조하여 작업(Work)을 배정하는 유닛을 프로세싱 블록(Processing block), 또는 서브코어(Sub Core)라고 부른다
- 위 사진을 보면 4개의 커다란 블록이 보일텐데 이 4개의 블록이 프로세싱 블록이다.
- 이 프로세싱 블록의 갯수는 GPU 아키텍처에 따라서 조금 차이를 보이지만 보통은 4개로 구성된다. 참고로 엔비디아 구형 아키텍처에서 SM은 2개의 프로세싱 유닛으로 구성되어 있었다
- 이러한 병렬 처리의 핵심은 공유 메모리(Shared Local Memory, 줄여서 SMEM 또는 SLM이라고 부른다) 기반의 데이터 전송인데, 같은 SM 내부에 있는 Processing block들은 SLM을 통하여 데이터를 주고받을 수 있기 때문에 SM 외부에 있는 GPC에서 데이터를 가져오거나 글로벌 메모리를 페치하는 것 보다 훨씬 빠른 속도로 처리할 수 있다.
- 또한 이 SLM 억세스는 충분히 빠르지만, 이 SLM 억세스를 최소화하는 작업도 필요한데 아무리 공유메모리가 빠르더라도 산술연산 대비 훨씬 많은 클럭 사이클을 소비하므로  이 접근 횟수를 최소화화는 것이 속도 향상으로 이어지기 때문이다.

#### 4.1. 스레드 블록 바리어 (Thread Block Barrier)

- SM은 데이터 공유기능 외에도 스레드 블록 단위의 동기화 기능을 제공한다.
- 쿠다 코어에서 ` __syncthreads()` 를 호출하면 스레드 블록에 포함된 모든 스레드의 동기화를 시도하는데, 이 동기화에 사용되는 하드웨어가 SM 내부에 있다

#### 4.2. 텐서 메모리 엑셀레이터(Tensor Memory Accelerator, TMA)

- 텐서 메모리 엑셀레이터는 2022년 호퍼 아키텍처에 추가된 하드웨어 유닛이다. 
- TMA는 이름이 시사하는 대로 메모리 읽기/쓰기 연산을 고속화하는 도구이다. 구체적으로 2가지 기능을 수행한다
- 첫째, GPC 내부의 SM들간에 대용량 데이터를 고속으로 교환하는 역할을 담당한다.
- 둘째, L2캐시 또는 전역 메모리에서 데이터를 로드할 때 고속으로 로드하는 기능을 수행한다

### 5. 프로세싱 블록 (Processing blocks)

![](/images/gpu_nvidia/h100-processing-block.jpg)

GH100 내부의 SM 구조. 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

- 프로세싱 블록은 다른 말로 서브코어(Sub-Core)라고도 불린다

- 이 프로세싱 블록은 산술 연산을 수행하는 하드웨어를 모아놓은 유닛이다. 이 산술연산 유닛을 스칼라 유닛 (Scalar Unit, 줄여서 SU)라고도 부른다

- 스칼라 연산 유닛의 종류와 갯수는 아키텍처별로 차이가 있지만 위의 GH100 아키텍처는 다음의 유닛을 가지고 있다
    - 16개의 INT32 전용 쿠다코어
    - 32개의 FP32(full-precision) 전용 쿠다코어
    - 16개의 FP64(double-precision) 전용 쿠다코어
    - 1개의 4세대 텐서코어

- 이 산술연산 유닛은 레지스터에서 데이터를 가져와 연산을 처리한 후에 다시 레지스터에 값을 저장하는 구조로 작동한다

- 각 레지스터는 32bit 크기로 모두 동일하며 스칼라 값을 저장한다 (int, float 등의 숫자값을 스칼라 값이라고 한다)

- GH100 아키텍처에서 레지스터의 갯수는 프로세싱 블록당 16,384개이다. 이 레지스터들은 레지스터 파일이라는 단위로 관리된다

- 이 16,384는 숫자는 512와 32를 곱한 결과이다. 즉 레지스터 파일은 512x32 크기의 타일 형태로 구성되어 있다

![](/images/gpu_nvidia/v100-register-file-v2.png)
(이미지 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ )

- 왜 512x32 크기의 타일이냐 하면 32는 병렬처리 단위인 워프(Warp)의 스레드 갯수이고, 512는 프로그램이 필요로 하는 레지스터의 갯수에 맞춰 스레드당 갯수가 가변적으로 정해진다
- 이 부분은 매우 헷갈리기 쉬운 부분이라 한번에 이해하기는 어렵다는 점을 미리 밝혀둔다
- 이 레지스터 부분을 제대로 이해하려면 워프(Warp)에 대한 이해가 선행되어야 한다
- 워프(Warp)는 실제로 존재하는 하드웨어가 아닌 소프트웨어적인 개념인데 이 워프 단위로 병렬처리가 수행된다. 엔비디아는 이 병렬처리를 SIMT(Single Instruction Multiple Threads)라고 명명하였다.
- 위 그림에서 확인할 수 있듯이 워프마다 할당받는 고유의 레지스터 블록이 있다. 그 레지스터 블록의 크기는 프로그램 코드에서 얼마나 많은 레지스터를 사용하는지에 따라 좌우된다
- 예를들어 프로그램에서 단지 62개의 레지스터만을 사용한다면 각 워프가 할당받는 레지스터의 크기는 62x32가 될 것이다. 
- 반면 프로그램에서 255개의 레지스터를 사용한다면 각 워프가 할당받는 레지스터의 크기는 255x32가 될 것이다. 255개는 이론적으로 사용가능한 레지스터의 최대 갯수이다 
- 그리고 프로그램에서 255개의 레지스터를 사용한다면 프로세싱 블록 하나당 2개의 워프(Warp)를 동시에 실행할 수 있다. 이 경우 레지스터 타일의 ROW는 510개의 ROW가 사용된다 (255 * 2 = 510)
- 이렇게 되면 레지스터 타일에서 맨 아래의 2개의 Row가 유휴(idle) 상태가 되는데 이 레지스터는 사용할 수 없다. 왜냐하면 어떠한 워프(Warp)에도 할당되지 않은 레지스터이기 때문이다




![](/images/gpu_nvidia/v100_gpu_processing_block.png)

h100 아키텍처의 프로세싱 블록. 빨간색 네모박스 안의 유닛이 프로세싱 블록이다

(이미지 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ )

- 이제 SM 관점에서 레지스터 블록을 살펴보자
- 위 그림을 보면 Warp 0, Warp1, Warp2, Warp3 ... 이런식으로 레지스터 블록이 할당된 것이 보일 것이다. 
- 즉 워프가 레지스터 블록을 할당받을 때, 하나의 프로세싱 블록에 모두 할당받는 것이 아닌 각 프로세싱 블록을 라운드 로빈 방식으로 순차적으로 순회하면서 할당받는 것을 확인할 수 있다
- 이건 직관적으로 생각해도 너무나 당연한데, 라운드 로빈 방식으로 워프를 할당해야 병렬 처리 기능을 극대화할 수 있기 때문이다.
- 예를들어 WMMA 명령어로 텐서코어 하드웨어를 사용해야 하는 상황이라고 가정하자. 현재 WebGPU에서는 텐서코어 유닛을 사용하는 것이 불가능하지만 텐서코어로 설명하는 것이 편하니 이것으로 설명하겠다.
- 각 프로세싱 블록당 보유하고 있는 텐서코어의 갯수는 1개이다. 
- 이런 상황에서 모든 워프를 하나의 프로세싱 블록에 할당하면 한개의 텐서코어를 놓고 2개 이상의 워프가 리소스를 공유해야 한다.
- 반면 라운드로빈 방식으로 워프를 각 프로세싱 블록에 할당한다면, 텐서코어 4개가 각 워프에 골고루 로드밸런싱 되므로 병렬처리 속도가 증가할 것임은 자명하다


#### 5.1 GPU내의 계층간 데이터 이동 속도의 차이

- 위의 레지스터 이미지를 참고로 하여 데이터 이동에 대해 살펴보겠다. 
- GPU내에 수십만개의 레지스터가 있지만 레지스터간의 데이터 이동 속도가 모두 같지는 않다
- 레지스터간의 하드웨어적인 거리가 근접할 수록 데이터 이동 속도가 더 빠르다. 구체적으로 말하면 다음과 같다

- 같은 워프에 속한 레지스터간의 데이터 공유가 제일 빠르다. 이들은 중간에 메모리를 경유하지 않고 레지스터와 레지스터간의 직접적인 데이터 전달이 가능하다. 이는 워프간에 데이터를 전달하는 특수 하드웨어를 통해 가능하다. 개발자는 워프 셔플(Warp Shuffle) 명령어로 이 기능을 구현할 수 있다. 쿠다 커널에서는 `__shfl_sync` 등으로 구현할 수 있고 WebGPU의 WGSL에서는 `subgroupShuffleDown` 등이 이 명령어에 해당한다. 상세는 [WGSL 공식 스펙](https://www.w3.org/TR/WGSL/#subgroupshuffledown-builtin)을 참조할 것.
- 같은 프로세싱 블록에 속한 레지스터간의 데이터 공유는 더 느리다. 이들은 비록 같은 프로세싱 블록에 속해있지만 SM의 공유메모리(Shared Local Memory,SLM) 를 경유해야 하기 때문이다
- 다른 프로세싱 블록에 속해있지만 같은 SM에 속해있는 레지스터간의 데이터 공유도 마찬가지로 SM의 공유메모리(Shared Local Memory,SLM) 를 경유해야 한다
- 같은 GPC에 속해있지만 다른 SM에 속한 레지스터간의 데이터 공유는 더 느리다. 2022년 공개된 호퍼(Hopper) 아키텍처부터 전역 메모리를 경유하지 않고 SM과 SM간의 데이터 공유가 가능해지기는 했다. 이를 `SM-to-SM Network` 라고 하는데 이는 GPC내부에 특수 설계된 SM과 SM과의 논리회로를 통해 가능해졌다. 이 회로는 SM이 다른 SM의 공유메모리(Shared Local Memory,SLM)를 읽을 수 있는 기능을 제공한다. 이를 분산 공유 메모리(Distributed Shared Memory, DSM) 라고도 한다. 엔비디아 측에 의하면 지연 시간이 기존 전역메모리 접근 대비 약 7배 이상 개선되었다.
- 다른 GPC에 속해있는 레지스터간의 데이터 공유는 더 느리다. 이들은 GPU의 L2캐시를 경유해서만 데이터 공유가 가능하기 때문이다. 만일 L2캐시 히트가 실패한다면 역메모리(VRAM)를 참조하는 상황이 발생한다
- 그리고 2024년 공개된 블랙웰 아키텍처는 다이 대 다이 (Die to Die) 전송이라는 레이어를 추가했다

![](/images/gpu_nvidia/blackwell-ultra.webp)

이미지 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

- 위 이미지는 블랙웰 울트라의 GPU 구조이다. 보시다시피 2개의 다이(DIE)를 연결하여 하나의 GPU로 만들었다
- 그리고 이 다이(Die)는 NV-HBI(NVIDIA High-Bandwidth Interface)라 불리는 하드웨어를 통하여 통신이 가능하다
- 이것은 블랙웰 아키텍처 이전에는 없던 개념이다. 왜 이런 구조를 추가했을지는 여러개의 GPU를 병렬로 사용하는 상황을 생각해보면 이해할 수 있다
- 아래 그림은 NV링크 기반의 서버용 하드웨어인데 4개의 GPU를 NV링크 슬롯에 꽂아두고 이를 병렬로 사용하는 시스템을 구축해 둔 것이다

![](/images/gpu_nvidia/nvlink.jpeg)
이미지 출처 : https://forums.developer.nvidia.com/t/nvlink-support-for-connecting-4-gpus/253975/7

- 만일 GPU간 데이터 전송이 필요하다면 NV링크 레인을 따라서 데이터를 전송해야 하므로 매우 긴 레이턴시를 피할 수 없다
- 그래서 GPU와 GPU간의 데이터 전송 속도를 더 높이려는 차원에서 GPU다이를 하나로 묶어 1개의 GPU로 만든 것이다
- 여기서 GPU다이간에 데이터 전송 용도로 사용되는 NV-HBI 인터페이스는 NV링크대비 더 빠른 속도를 제공한다


### 6. 워프 (Warp)

- 워프는 하드웨어 유닛이 아닌 가상의 병렬 처리 단위이다.
- 32개의 스레드를 묶은 최소한의 병렬 처리 단위이다.
- 당신이 어떤 프로그램을 작성했는데 그 프로그램이 설명 1개의 스레드만을 요구한다고 하더라도 1개의 Warp를 점유하여 실행된다.
- 이렇게 하면 나머지 31개의 스레드는 유휴 상태가 되므로 자원의 낭비라고 생각할 수 있지만 어쩔 수 없다. 왜냐하면 앞서 언급한 대로 Warp는 최소한의 병렬 처리 단위이기 때문이다
- 그리고 GPU에서 실행되는 모든 스레드는 그것이 설령 1개의 스레드만을 요구하더라도 병렬 처리 단위로 수행된다
- 이러한 병렬 처리는 모든 스레드가 동일한 명령어를 실행하되, 각자 다른 데이터를 처리하는데 이를 SIMT(Single Instruction, Multiple Thread)라고 부른다. 이는 CPU에서 작동하는 SIMD와 매우 유사한 컨셉으로 작동한다
- Warp 스케줄러가 레지스터/메모리 준비 상태에 따라 실행할 warp를 선택한다 (이게 무슨 말이지 ? 그건 바로 : 필요한 데이터가 L1 캐시나 Shared Memory에 존재하는지 여부에 따라 실행할 WARP를 결정한다 ?)

#### 6.1. 쿠다코어 (CUDA Core) 

![](/images/gpu_nvidia/cuda_core.png)

사진 : nvidia Fermi 아키텍처의 쿠다코어

출처 : https://www.researchgate.net/figure/FIGURE18-Overview-of-typical-Graphics-Processing-Unit-GPU-architecture-a_fig2_342826318

쿠다 코어는 fma(Fused Multiply Add)라는 산술연산에 최적화된 하드웨어이다. 텐서코어가 등장하기 전 까지는 AI학습과 추론의 행렬곱은 이 쿠다 코어에 의해 수행되었다.

#### 6.2. 텐서코어(Tensor Core)

- 텐서코어는 행렬곱 전용 하드웨어이다. 2017년 출시된 Volta 아키텍처에 처음 도입되었다
- WMMA (Warp Level Matrix Multiply Accumulate)라고 불리는 PTX 명령어가 바로 이 텐서코어에서 실행된다
- 안타깝게도 현재 2025년 기준으로 WebGPU의 WGSL로는 텐서코어 하드웨어를 사용하는 명령어를 컴파일하는 것이 불가능하다.
- 따라서 텐서코어를 사용하려면 CUDA나 다른 저수준 API를 사용해야 한다

![](/images/gpu_nvidia/h100-tensor-core.png)
사진 : A100과 H100의 텐서코어 모형도. 출처 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

[TODO]위 그림처럼 텐서코어를 설명하는 이미지는 모두 3차원으로 표현되어 있다. 실제 GPU 다이는 2차원 평면에 반도체로 구현되어 있는데 왜 텐서코어를 설명하는 이미지는 모두 3차원인가 ?
<!-- answer here -->

### 7. 끝 

이상으로 엔비디아 GPU 하드웨어에 대한 소개가 마무리되었다

---

# 참고자료

- Thread Block Clusters 관련 : https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Thread Block Clusters 관련 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- GPU System Processor 관련 정보 : https://www.glennklockwood.com/garden/GSP
- 텐서 메모리 엑셀레이터(Tensor Memory Accelerator, TMA) 관련 : https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth
- [Nvidia Blog : Turing Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)
- [Nvidia Blog : Hopper Architecture in Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Nvidia Blog : Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [Stanford Seminar - Nvidia’s H100 GPU By Jack Choquette, Nvidia](https://www.youtube.com/watch?v=MC223HlPdK0)
- [Tensor Core Lecture by NVIDIA CUTLASS Team ](https://www.youtube.com/watch?v=hQ9GPnV0-50)