---
title: 인텔 GPU 최적화 - 정적 메모리 coalescing으로 속도 최적화하기
published: 2025-12-19
description: ""
tags: []
category: Default
draft: false
---

<!-- 이 글에서 부족한 점 : 최하단의 opencl로 작성된 코드가 실제로 어떻게 컴파일되는지 정보가 없음. 실제 컴파일을 해볼것 -->
<!-- 문제 : 그래서 problem을 제시했으면 solution도 제시해야 하는데 solution이 부족함. i2o_a 함수의 대안을 제시하지 못하였다. -->

## 대상 독자 

이 글은 아래의 지식을 어느정도 알고 있다는 전제로 작성되었다
- 인텔 GPU 아키텍처에 대한 기본 지식
- WGSL(WebGPU Shader Language) 문법에 대한 기본 지식

## 글의 목표 

제목처럼 작성한 코드를 정적 메모리 coalescing이 적용되도록 최적화하는 것이다.

먼저 정적 메모리 coalescing (Static Memory Coalescing)이 무엇인지 설명하겠다.

`coalesce`를 사전에서 검색하면 `(더 큰 덩어리로) 합치다`라는 의미인데 여러개의 메모리 주소를 하나로 합쳐서 메모리 억세스 횟수를 줄이는 기법을 의미한다.

예를 들어 2번의 메모리 읽기를 수행한다고 가정하자. 만일 이 2개의 메모리 주소가 인접하지 않고 멀리 떨어져있다면 2번의 메모리 읽기를 피할 수 없다. 하지만 이 2개의 메모리 주소가 서로 인접한 주소일 경우 이론적으로 1번의 메모리 읽기로 줄일 수 있다. 즉 2개의 메모리 주소를 하나의 덩어리로 합쳐서 읽는 것이다. 이것으로 메모리 억세스 횟수를 줄일 수 있으므로 속도 최적화에 도움이 된다.

이렇게 `Memory Coalescing`의 의미는 알겠지만 정적(Static)이라는 단어가 붙는 이유는 무엇인가 ? 

이는 컴파일 타임에 메모리 주소가 확정되는 경우를 의미한다. 즉 컴파일러가 컴파일 타임에 메모리 주소가 연속된 주소라는 것을 알아차려 컴파일된 코드에 `Memory Coalescing` 정보가 포함되어 있는 것을 의미한다.
그리고 Static 방식이 있으니 그 반대인 Dynamic 방식도 있을텐데 Dynamic 방식은 런타임에 `Memory Coalescing`이 처리되는 경우를 의미한다. 즉 컴파일 타임에는 메모리 주소가 연속된 주소인지 아닌지 알 수 없지만 런타임에 들어가서 실제 메모리 주소가 연속된 주소라는 것을 알아차리는 경우이다. 
다시 말해서, 개발자가 코드 최적화를 수행하건 수행하지 않건 `Memory Coalescing` 최적화는 발생한다 (로직을 대대적으로 개편한 경우는 제외한다). 다만 그 최적화가 컴파일 타임에 수행되는지, 런타임에 수행되는지의 차이가 있을 뿐이다. 
만일 런타임에 `Memory Coalescing`이 수행된다면 동적으로 메모리 주소가 연속된 주소인지 판단하는 과정이 추가되므로 이에 따른 오버헤드가 발생하여 속도가 느려진다. 그러므로 가급적이면 정적 방식의 메모리 Coalescing을 시도하는 것이 나을 것이다

## Static Memory Coalescing 여부를 확인하는 방법

작성한 코드에 최적화가 적용되었는지 여부를 확인하는 가장 확실한 방법은 컴파일된 코드를 확인하는 것이다.

이 글에서는 WGSL(WebGPU Shader Language)로 작성된 코드를 인텔 GPU용 머신 코드로 컴파일하고 그 코드를 분석하는 방법을 소개한다.

구글에서 만든 [Dawn](https://github.com/google/dawn) 이라는 WebGPU 구현체가 있는데 이 Dawn에 포함된 WGSL 컴파일러인 Tint 사용하여 SPIR-V로 컴파일할 수 있다. 구체적인 tint를 사용법은 dawn 레포지토리를 참조하라.

그렇게 컴파일된 SPIR-V는 일종의 중간언어 표현(Intermediate Representation, IR)인데 기계어로 최종 컴파일하기 직전단계이다. tint 컴파일러가 수행하는 것이 바로 WGSL 코드를 SPIR-V로 컴파일하는 작업이다. 이 단계가 필요한 이유는 여러 종류의 프론트엔드 코드(예 : WGSL, OpenCL, CUDA)와 최종 GPU 머신 코드(예 : 인텔 GPU, 엔비디아 GPU, AMD GPU)를 연결하는 공통 인터페이스 역할을 하기 때문이다. 이런 공통의 인터페이스가 있으면 다양한 프론트엔드 언어와 GPU 아키텍처를 지원하는 것이 용이해진다는 이점이 있다.

정리하면 컴파일 프로세스는 다음과 같다. 

`WGSL(WebGPU Shader Language) -> SPIR-V(IR) -> 인텔 GPU 머신 코드 -> 머신 코드를 디스어셈블 하여 인간이 읽을 수 있는 형태로 표현`

tint 컴파일러를 사용하여 SPIR-V로 컴파일하고, 그 SPIR-V 코드를 입력으로 받아 Vulkan API에게 전달하여 인텔 GPU용 머신 코드로 컴파일한다. Vulkan API를 활용하여 GPU 머신 코드로 컴파일하는 과정은 제미나이 등의 LLM에게 요청하면 어렵지 않게 구현할 수 있다.

그렇게 컴파일하면 아래와 같은 코드를 발견할 수 있다

```text
0x00000170: send(16) g23UD  g29UD  nullUD  0x086458fd  0x00000000
hdc1 MsgDesc: (DC A64 untyped surface read, Surface = 253, SIMD16, Mask = 0x8)  mlen 4 ex_mlen 0 rlen 6 { align1 1H $0 };
```

위의 코드는 MatMul을 수행하는 WGSL(WebGPU Shader Language) 코드를 인텔 GPU에서 실행되는 머신 코드로 컴파일한 결과인데 메모리 읽기 작업을 수행한다. 이 코드를 해석할 수 있으면 정적 메모리 coalescing이 적용되었는지 여부를 판단할 수 있다. 미리 말해두지만 위 코드는 정적 메모리 coalescing이 적용되지 않았다

왜 이런 최적화 점검이 필요할까? matmul 함수에서 속도가 저하되는 지점은 의외로 행렬곱 연산 그 자체보다는 메모리 읽기/쓰기 시점에 발생하기 때문이다. 이를 메모리 바운드라고 한다. 그러므로 컴파일된 `send` 명령어를 참고하여 정적 메모리 Coalescing 여부를 점검하여  최적화를 수행하는 것은 속도 향상에 큰 이점을 준다

## 코드 분석 (send 명령어의 첫번째줄)

이제 본격적으로 코드를 분석하자. 위 코드의 첫번째 줄은 다음과 같다

```
0x00000170: send(16)  g23UD  g29UD  nullUD  0x086458fd  0x00000000
```

`0x00000170` : 이 명령어가 저장된 메모리 어드레스를 의미한다. 주소값이 0x170으로 매우 작은 이유는 가상 메모리 주소이기 때문이다.

`send(16)` : 데이터 전송과 관련된 Opcode(operation code)이다. 설명을 쉽게 하려고 위에서 메모리 읽기/쓰기 작업을 한다고 적어었지만 사실 부정확한 설명이다
send 명령어는 `external function units`에 데이터를 전송하는 기능을 수행한다
`external function units`는 용어가 생소할 수 있지만 이 개념을 이해하려면 아래 그림을 참조하라

![](/images/intel_gpu/intel-integrated-GPU-architecture.png)
(출처 : https://www.researchgate.net/figure/ntel-integrated-GPU-architecture_fig2_346030522)

위 그림의 execution unit(EU) 외부를 `external`이라 부르고 외부에 있으면서 특수 기능을 수행하는 유닛을 `function units`이라 부른다
그러므로 EU가 EU 외부의 하드웨어인 `function unit`과 소통할 때 send 함수를 호출한다
그리고 그 대표적인 function unit이 그림에 있는 `Data Port`인데 이 `Data Port`를 경유하여 Shared Local Memory(SLM) 또는 VRAM과 소통할 수 있다.
메모리를 포함하여 샘플러, URB(유니파이드 반환 버퍼) 등의 모든 external function units과 소통할 때 Data Port를 경유한다

한가지 SLM에 대해 헷갈릴 수 있는데 인텔 12세대 GPU부터 SLM의 위치가 변경되었다는 것이다. 인텔 11세대 까지의 GPU에서 SLM은 `Subslice` 외부에 위치하였기 때문에 `Data Port`를 경유하여 `Subslice` 외부 회로로 나가서 접근할 수 있었다

아래는 인텔 8세대 GPU 아키텍처인데 Data Port를 통하여 외부의 SLM으로 접근하는 회로를 보여준다 .

![](/images/intel_gpu/IntelGen8GPU-fs8.png)
(출처 : The Compute Architecture of Intel® Processor Graphics Gen8 의 11페이지)

그러다가 인텔 12세대에 접어들면서 SLM의 위치가 Subslice 내부로 이전되었다.아래 그림 오른쪽 하단에 SLM이 Subslice 내부에 위치한 것을 볼 수 있다.

![](/images/intel_gpu/intel-xe-subslice.jpg)
출처 : [Jeffrey Burt by The Next Platform](https://www.nextplatform.com/2020/09/02/intel-puts-its-xe-gpu-stakes-in-the-ground/)

이처럼 SLM의 위치가 Subslice 내부로 이전된 것은 SLM과의 물리적 거리를 줄여 레이턴시를 줄이기 위해서이다.

`(16)` : `send(16)` Opcode의 서브옵션(sub-option)이다. `(16)`은 SIMD 레벨을 의미하는데 즉 SIMD16 모드로 WGSL 함수가 실행됨을 의미한다.
SIMD16 모드란 같은 코드를 16개의 스레드가 동시에 병렬로 실행한다는 뜻이다. WGSL에서는 subgroup이 16으로 설정될 때 SIMD16 모드로 컴파일된다 
(주의 : 이 부분은 추가 확인이 필요하다). 
즉 인텔 GPU의 SIMD 모드는 서브그룹 단위로 실행되는 경향이 있다

`g23UD` : 세번째로 등장하는 `g23UD`는 데이터를 저장할 레지스터를 의미한다.
G는 `general register`의 약자이며 23은 레지스터의 index 번호이다.
각 physical thread는 128개의 레지스터를 가지고 있는데 그 중 인덱스가 23번째인 레지스터라는 뜻이다.
뒤에 붙은 UD는 데이터 타입인데 Unsigned Doubleword의 줄임말이다.

`g29UD` : 네번째로 등장하는 `g29UD` 레지스터는 첫번째 소스 레지스터를 가리키는데 VRAM의 메모리 주소를 담고있다. 이 메모리 주소에서 데이터를 가져온다
이를 페이로드(Payload) 또는 메시지라고 부른다. 이 값은 data port에 전달되는 일종의 argument이므로 이를 해석할 추가 정보가 필요하다. 이는 아래에서 설명하겠다.

`nullUD` : 다섯번째로 등장하는 nullUD는 값이 null인 만큼 사용되지 않는다. 다른 명령어에는 두번째 소스 레지스터가 표기된다.

`0x086458fd` : 여섯번째로 등장하는 이 메모리 주소는 본인도 모른다 (TODO : 보강할 것)

`0x00000000` : 일곱번째로 등장하는 메시지 디스크립터인데 여기서는 사용되지 않는 것으로 보인다 (TODO : 보강할 것)

여기까지가 명령어에 대한 분석이었고 그 아랫줄은 메시지(Message)에 대한 설명을 보여준다

## 코드 분석 (send 명령어의 두번째줄)

```
hdc1 MsgDesc: (DC A64 untyped surface read, Surface = 253, SIMD16, Mask = 0x8)  mlen 4 ex_mlen 0 rlen 6 { align1 1H $0 };
```

`hdc1` : Hardware device cache 1의 줄임말이며 data port를 의미한다.

`MsgDesc` : message description의 줄임말이다. 즉 `hdc1 MsgDesc : ( ... )`로 표현되는 이 문장은 data port에 전달된 메시지에 대한 상세 정보는 이러이러하다는 세부 정보를 data port에 전달하여 data port가 이 데이터를 어떻게 해석하여 어떤 행동을 수행할지를 명시한다 

이어지는 `DC A64 untyped surface read`는 전달된 메시지의 속성을 나타낸다

`DC` : Data cache access (what it means?)

`A64` : 64bit (stateless) address를 의미한다. 이 명령어는 64비트 가상 메모리 포인터를 사용한다는 것을 의미한다. 즉 위에 등장하는 `g29UD` 레지스터를 64비트 가상 메모리 주소로 해석하겠다는 것을 의미한다

`untyped` : 전달된 데이터를 후처리 변경 없이 raw data로 읽겠다는 것을 의미한다 (untyped가 아니라 unsigned doubleword가 아닌가?)

`surface` : surface는 data port의 명령어를 의미하는데 data port는 총 256가지의 명령어를 수행할 수 있다

`read` : 외부 장치에서 데이터를 읽어 레지스터에 저장한다는 것을 의미한다

`Surface = 253` : 253번 Surface 명령어는 VRAM에서 데이터를 읽어들이는 것을 의미한다
인텔 하드웨어에서 253번 인덱스는 Stateless 모델로 reserve되어 있다. 이는 쉐이더가 VRAM에 64비트 포인터로 접근할 수 있음을 의미한다
(Stateless A64 addressing (random access))

`SIMD16` : 16개의 가상 스레드가 병렬로 수행되는 것을 의미한다. 다시말해 16회의 읽기 작업이 요청되지만 Data Port 내부에 있는 L1(Level 1) 캐시 컨트롤러에서 읽기 최적화 작업을 수행하여 최대한 적은 횟수로 메모리 읽기 작업을 수행하도록 시도한다.

`Mask = 0x8` : Untyped scattered read operation 에서 사용되는 마스크 비트이다.

`mlen 4` : 메시지(message)의 길이가 4라는 의미이다. 즉 억세스할 메모리 주소를 표현하는데 register가 4개가 필요하다는 의미가 된다
왜 register가 4개가 필요한지 계산해 보자. 레지스터 한개당 32byte를 저장할 수 있는데 bit로 환산하면 32\*8=256bit가 된다. 즉 레지스터 한개당 64bit 메모리 주소를 4개 표현할 수 있다. 그런데 위에서 설명했듯이 이 명령어는 SIMD16 명령어로 수행되므로 16개의 메모리 주소가 필요하며 그렇기 때문에 레지스터 4개를 동원하여 16개의 메모리 주소를 나타낸 것이다

`ex_mlen 0` : extra message length의 줄임말이며 이 코드에서는 사용되지 않는다

`rlen 6` : return length 6의 약자로서 리턴값을 표현하는데 총 6개의 레지스터가 필요하다는 것을 의미한다. 6개의 레지스터는 6*32Byte = 192Byte = 1536bit의 정보를 담을 수 있다. 이를 스레드 갯수인 16으로 나누면 96bit이므로 각 스레드별로 96bit 데이터를 리턴한다는 뜻으로 해석할 수 있다.

`{ ... }` : 중괄호 안에는 추가적인 메시지 옵션이 포함된다

`align1` : align은 register access alignment mode의 줄임말이며 레지스터 내에서 데이터가 어떻게 배치되는지를 나타낸다. `align1`은 레지스터 내에서 데이터가 "수평(horizontal)" 패턴으로 배치됨을 의미한다. 본 예제에서는 SIMD16으로 실행되는 16개의 스레드가 4개의 레지스터(g29, g30, g31, g32)를 공유한다. 각 레지스터는 32바이트(256비트)이므로 4개의 64비트 메모리 주소를 담을 수 있다. 16개의 스레드는 이 4개의 레지스터를 공유하며, 각 스레드는 레지스터 내에서 자신의 독립적인 데이터 슬라이스에 접근한다. 예를 들어, 스레드 0은 g29의 0~63비트, 스레드 1은 g29의 64~127비트를 사용하는 식이다

`1H` : 1 half의 줄임말이며 SIMD32를 full width라고 가정했을 때 그 절반 크기인 SIMD16 모드로 실행하겠다는 것을 의미한다

`$0` : 스코어보드의 인덱스가 0번이라는 의미이다
SEND 명령어가 처음 실행될 때 이 스코어보드는 BUSY로 설정된다. SEND 작업이 수행중이라는 뜻이다
SEND 작업이 완료되면 게이트웨이는 Release 시그널을 보내어 스코어보드의 값을 변경한다. 이 때 스코어보드의 값은 클리어된다(리셋된다)
이 스코어보드의 값이 리셋 유무를 판정하여 해당 작업이 완료되었는지를 확인하고 다음 코드를 진행할 수 있다

이상으로 send 명령어에 대한 설명이 종료되었다. 

---

-> 이 지식을 어떻게 응용할 수 있을까 
-> Matmul 함수를 작성한뒤 컴파일을 수행하여 send 명령어가 어떻게 컴파일 되었는지 확인하는 것이다. 최적화가 수행되었는지, 최적화가 되지 않았다면 내 코드의 어떤 부분이 문제인지 분석하여 그 부분을 고치고 다시 컴파일하여 최적화가 수행되었는지 반복된 피드백 루프를 통하여 코드를 최적화할 수 있다

-> 구체적인 방법으로는, 메모장을 열고 `Surface = 253`으로 검색하면 VRAM에 대한 읽기/쓰기 명령어를 찾을 수 있으므로 복잡한 컴파일 코드를 처음부터 읽을 필요가 없다. 그렇게 해서 `mlen`의 길이가 몇인지를 체크하면 되는것인데 위의 예제처럼 4라면 16번의 메모리 읽기 작업이 요청되므로 최적화에 실패했다는 것을 알 수 있다. 그리고 가능하다면 코드를 수정하여 mlen의 길이를 줄일 수 있는데 이는 정적 메모리 Coalescing이 적용되었다는 뜻이므로 메모리 바운드 시간을 유의미한 수준으로 줄일 수 있다

-> 이런 전략은 Shared Local Memory에도 비슷하게 응용할 수 있다. 예를 들어 `Surface = 254`라고 적힌 Send 명령어는 SLM에 대한 읽기/쓰기 명령어이므로 SLM에 대한 접근이 최적화 되었는지를 비슷한 방식으로 확인할 수 있다

## 정적 메모리 Coalescing을 적용하여 레이턴시를 낮추기

컴파일러가 정적 메모리 Coalescing을 적용하도록 유도하려면 까다로운 조건을 만족시켜야 한다. 즉 16개의 주소가 연속된 주소라는 것을 컴파일러가 컴파일 타임에 알아차리고 이를 한번의 연속된 메모리 억세스로 최적화하도록 유도해야 한다. 

다시 말해 위의 코드는 컴파일 타임에 16개의 메모리 주소가 연속된 주소라는 정보를 컴파일러가 캐치하지 못한 것이다. 그래서 위와 같이 16번의 별개의 읽기 요청으로 컴파일된 것인데 이런 메모리 접근을 `Scattered load`라고 한다.

왜 메모리 읽기 최적화에 실패했는지 실제 코드를 보면서 이해해보자 


```wgsl

enable f16;
enable subgroups;

struct Uniforms { a_shape:vec3<u32>, a_strides:vec3<u32>, b_shape:vec3<u32>, b_strides:vec3<u32>, scales_shape:u32, scales_strides:u32, output_shape:vec3<u32>, output_strides:vec3<u32> };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// 컴파일 타임에 이 i2o_a 함수의 리턴값이 연속된 메모리 주소라는 것을 
// 확신할 수 없으므로 스캐터드 로드가 발생한다
fn i2o_a(indices: vec3<u32>) -> u32 {
    return uniforms.a_strides[2] * (indices[2])+uniforms.a_strides[1] * (indices[1])+uniforms.a_strides[0] * (indices[0]);
}

fn get_aByIndices(indices: vec3<u32>) -> vec4<f32> {
    // a라는 행렬에 접근하는 코드. 여기서 실제적인 메모리 읽기 작업이 발생한다
    return a[i2o_a(indices)];
}

var<workgroup> sub_a: array<vec4<f32>, 128>;
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
          @builtin(local_invocation_id) local_id : vec3<u32>,
          @builtin(local_invocation_index) local_idx : u32,
          @builtin(workgroup_id) workgroup_id : vec3<u32>,
          @builtin(num_workgroups) num_workgroups : vec3<u32>) {

         let workgroup_index = workgroup_id.z * num_workgroups[0] * num_workgroups[1] + workgroup_id.y * num_workgroups[0] + workgroup_id.x;
         let global_idx = workgroup_index * 128u + local_idx;

          let output_indices = o2i_output(workgroup_index * 8);
          let col = output_indices[2];
          let row = output_indices[1];
          let batch = output_indices[0];
          let n_blocks_per_col = uniforms.b_shape[1];
          let num_tiles =  (n_blocks_per_col - 1) / 16 + 1;

          var local_sum: f32 = f32(0);

          // Loop over shared dimension.
          for (var tile: u32 = 0; tile < num_tiles; tile += 1) {
            let a_col_start = tile * 128;

            // load one tile A data into shared memory.
            for (var a_offset = local_idx; a_offset < 128; a_offset += 128)
            {
              let a_col = a_col_start + a_offset;
              if (a_col < uniforms.a_shape[2])
              {
                // 이 부분에서 메모리 읽기 작업을 시도한다
                sub_a[a_offset] = get_aByIndices(vec3<u32>(batch, row, a_col));
              } else {
                sub_a[a_offset] = vec4<f32>(0);
              }
            }
            // 이하생략
         }
}

```

이 코드는 학습을 위한 예제 코드가 아니라 `onnxruntime-web`의 matmul 커널에서 실제로 사용되는 코드이므로 큰 의미가 있다.

위의 코드에서 VRAM 데이터를 레지스터로 복사하는 아래 대목에 주목하라.

```wgsl
sub_a[a_offset] = get_aByIndices(vec3<u32>(batch, row, a_col));
```

위 코드는 아래 함수로 이어진다

```wgsl
fn get_aByIndices(indices: vec3<u32>) -> vec4<f32> {
    // a라는 행렬에 접근하는 코드. 여기서 실제적인 메모리 읽기 작업이 발생한다
    return a[i2o_a(indices)];
}
```

그리고 또다시 아래 함수로 이어진다. 

```wgsl
fn i2o_a(indices: vec3<u32>) -> u32 {
    return uniforms.a_strides[2] * (indices[2])+uniforms.a_strides[1] * (indices[1])+uniforms.a_strides[0] * (indices[0]);
}
```

이 함수가 바로 문제의 핵심이다. `uniforms.a_strides`와 `indices` 값이 런타임에 결정되므로 컴파일 타임에 이 함수의 리턴값을 확정할 수 없다. 

그렇기 때문에 a 행렬의 인덱스를 확정하지 못하는 것이고, 메모리 주소를 확정하지 못한다. 이것이 정적 메모리 Coalescing이 적용되지 않는 원인이다.

그렇다면 어떻게 코드를 작성해야 컴파일러 최적화가 수행되는지 인텔 공식 예제로 알아보자

```cpp
constexpr int N = 1024 * 1024;
int * data = sycl::malloc_shared < int > (N, q);
int * data2 = sycl::malloc_shared < int > (N, q);
memset(data2, 0xFF, sizeof(int) * N);

auto e = q.submit([ & ](auto & h) {
  h.parallel_for(sycl::nd_range(sycl::range { N / 16 }, sycl::range { 32 }),
    [ = ](sycl::nd_item < 1 > it) {
      int i = it.get_global_linear_id();
      auto sg = it.get_sub_group();
      int sgSize = sg.get_local_range()[0];
      i = (i / sgSize) * sgSize * 16 + (i % sgSize);
      for (int j = 0; j < sgSize * 16; j += sgSize) {
        data[i + j] = data2[i + j];
      }
    });
});
```
(출처 : oneAPI GPU Optimization 37페이지 )

This kernel copies an array of 1024 x 1024 integers to another integer array of the same size. Each workitem copies 16 contiguous integers. However, the reads from data2 are gathered and stores to data are scattered. 

It will be more efficient to change the code to read and store contiguous integers in each subgroup instead of each work-item.

OpenCL로 작성된 위의 코드는 컴파일 타임에 a[i]의 인덱스가 확정된다. a의 인덱스는 0,1,2,3이므로 메모리 접근 대상은 a[0], a[1], a[2], a[3] 이며 컴파일러는 이를 연속된 메모리 주소로 인식하여 하나의 메모리 억세스로 최적화할 수 있다. 

그래서 정적 메모리 Coalescing은 다이나믹 메모리 Coalescing보다 얼마나 빠른 것인가 ?

Scattered Read(16개의 주소)를 기반으로 메모리 억세스를 시도할 때, 하드웨어 레벨에서 이를 감지하여 통합(Coalescing) 과정을 거치는 이 시간만큼 속도차이가 발생할 것이다. 이 부분은 확인이 필요하지만 대략 2~3배 정도 빨라지는 것으로 보인다.