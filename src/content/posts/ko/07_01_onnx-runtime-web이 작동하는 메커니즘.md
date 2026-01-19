---
title: Onnxruntime-web 실행흐름 정리
published: 2025-10-12
description: ""
tags: []
category: Default
draft: true
---

1단계 : 모델을 생성한다

inferencesession 인스턴스를 생성한다 

2단계 : inferencesession.run() 메서드를 호출한다 

3단계 : 이 코드는 onnx-runtime-web 라이브러리 내부에서 WebAssembly 모듈을 호출하여 실행된다

이제 wasm 내부로 진입했다. 여기서부터는 onnx-core 로직이 작동하는 것이다

즉 onnx는 본래 c++로 작성된 라이브러리이다. 이 c++ 코드를 웹에서 실행가능하도록 WebAssembly 모듈로 변환한 것이 onnx-runtime-web 라이브러리이다

코어 로직을 수행하던 중 백엔드가 webgpu임을 발견한다

webgpu 백엔드를 수행하려면 다시 js에게 제어권을 넘겨준다 

js로 구현된 webgpu 백엔드는 그래프를 해석한다

그래프를 해석하고 node를 만든다 

각 node는 program을 가지고 있다. 이 program은 WGSL 코드이다

이 프로그램은  정적 STATIC 코드가 아니라 추론을 할 때 동적으로 생성된다

예를들어 데이터타입이 F32인지 f16인지 여부를 코드를 생성할 때 결정한다