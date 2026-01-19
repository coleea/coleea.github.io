---
title: avif 컨버터 구현기
published: 0001-01-01
description: ""
tags: []
category: Default
draft: true
---

# 나 자신한테 물어보라. 과연 이 프로그램이 무슨 가치가 있지?

프로젝트 경로

G:\my\avif-converter-rust-wasm-vite

## 작동 프로세스 설명

- 핵심 로직은 다음과 같다
 
- 유저로부터 이미지 파일을 입력받는다
- File 인스턴스를 ArrayBuffer로 변환한다
- UInt8Array로 포멧을 변환한다
- UInt8Array로 포멧의 이미지를 wasm으로 전달한다
- 전달된 이미지는 ravif(rust-avif) 라이브러리로 이동된다
- ravif 내부에서 이미지를 디코딩한다. 디코딩하면 Vec<RGBA8> 데이터가 생성된다. 8비트 RGBA 포멧이라는 의미이다.
- 8비트 RGBA 포멧의 이미지를 avif 포멧으로 인코딩한다. 
- 이 인코딩 작업은 rav1e라는 라이브러리가 담당한다
- 즉 avif 인코딩의 코어 로직은 rav1e가 담당하고 있다
- rav1e의 알고리즘을 파는것은 상당한 시간소비를 요구한다

# rav1e가 실행되는 과정

- 먼저 컨텍스트(Context)를 생성한다
- 그리고 send_frame 메소드를 호출한다. 이렇게하면 worker스레드에서 작동하는 rav1e context가 이 프레임을 전달받는다
- 그리고 flush()메소드를 호출하면 워커 스레드에서 작동하는 rav1e에게 `이제 인코딩 작업을 수행하라`는 메시지를 보내고 이때부터 인코딩이 수행된다.
- rav1e의 context 는 context_innernals(`internals.rs`)에게 작업을 위임한다. `internals.rs`는 다시 `encoder.rs`의 `encode_frame`에게 작업을 위임한다.


# rav1e 성능 최적화하기

- rav1e의 특정 부분을 wgsl로 포팅하여 병렬처리 성능을 극대화하고 싶다
- 그 로직중 SATD 관련 로직이 포팅에 적합한 로직이다
- 실행 흐름은 아래와 같다
- receive_packet -> compute_block_importances -> update_block_importances -> get_satd -> hadamard8x8 -> hadamard2d -> hadamard8_1d
- wgsl 코드를 구현하기에 앞서 이 코드가 실제로 시간을 얼마나 소비하는지를 확인하는 작업이 선행되어야 할 것이다.


# 로직의 문제점

- 대용량 이미지를 인코딩하는 속도가 느리다. 인코딩 작업이 CPU 의해 수행되기 때문이다.
- 예를들어 8000x8000 이미지를 인코딩하는 것은 거의 1분 이상의 시간을 요구한다
- 왜 이렇게 느린가? 아마도 cpu-based 인코딩이라 시간이 오래 걸릴 것이다.
- 이것이 사람들의 한계이다. 대부분의 로직을 CPU로만 작성할 줄 알지, 컴퓨트 쉐이더로 작성하여 병렬 처리하는 개념에 대해 모른다
- 즉 완전히 똑같은 로직이라도.. WGSL로 만들어서 쿠다 코어를 사용하면 속도가 비약적으로 상승할 것임은 자명하다.
- 이 알고리즘의 핵심은 `wasm-by-rust\ravif\src\av1encoder.rs` 파일에 있는 `encode_raw_planes_internal` 함수일 것이다. 이 함수를 많이 수정한 흔적이 내 코드에 그대로 있다
- 왜 이 로직을 WGSL로 변경할 엄두가 나지 않냐면, 모든 코드를 WGSL로 이식할 생각을 했기 때문이다. 하지만 그럴 필요가 없다
- 병렬처리가 가능한 핵심 로직만을 WGSL코드로 변경하면 되는 것이기 때문이다.

## 러스트 코드를 컴파일하여 wasm 파일을 얻고, 이 wasm 파일을 vite 프로젝트에 추가하여 실행하는 방법