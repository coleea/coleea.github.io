---
title: nsight compute로 gpu 커널 프로파일링 하기
published: 0001-01-01
description: ""
tags: []
category: Default
draft: true
---

우분투 기준으로 설명한다
터미널에서 nsight compute 명령어인 `ncu --version`를 실행하여 정상 설치었는지 확인한다
정상 설치되었으면 실행파일 앞에 ncu 명령어를 추가하여 프로파일 결과를 저장할 수 있다

예를 들어 speech-to-text 프로램인 whisper를 아래의 명령어로 실행한다고 가정하자 
`whisper-cli -m models/ggml-base.bin  -f sample.wav` 
위 프로그램의 일부는 cuda 커널로 작성되어 있기 때문에 실행도중 발생하는 로그를 nsight compute로 수집하는 것이 가능하다. 구체적으로 아래와 같이 입력하여 데이터를 수집할 수 있다 

`ncu -o kernel_analysis_output --set basic whisper-cli -m models/ggml-base.bin  -f sample.wav`
 위 명령어로 실행하면 일반 실행시에는 보이지 않던 메시지가 터미널에 표시될 것이다. 구체적으로 아래와 같다

```
==WARNING== No metrics to collect found in sections.
==PROF== Connected to process 325185 (/home/lkb/proj-oss/whisper.cpp/build/bin/whisper-cli)
==PROF== Profiling "im2col_kernel" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "Kernel2" - 1: 0%....50%....100% - 1 pass
==PROF== Profiling "convert_unary" - 2: 0%....50%....100% - 1 pass
==PROF== Profiling "k_bin_bcast" - 3: 0%....50%....100% - 1 pass
==PROF== Profiling "unary_op_kernel" - 4: 0%....50%....100% - 1 pass
==PROF== Profiling "im2col_kernel" - 5: 0%....50%....100% - 1 pass
==PROF== Profiling "Kernel2" - 6: 0%....50%....100% - 1 pass
==PROF== Profiling "convert_unary" - 7: 0%....50%....100% - 1 pass
```

이 메시지들은 커널을 프로파일링 하고 있다는 메시지를 표시한 것인데 예를 들어 위 메시지에서 4번째 줄의 `im2col_kernel` 은 함수 이름이다. 트랜스포머로 구현된 프로그램은 수천번의 커널 호출이 발생하므로 프로파일링에 상당한 시간이 소요된다. 위 예제에서는 5675-2번의 커널 호출이 발생하였다 


 프로파일링이 종료되면 아래의 메시지가 표시된다. 

```
==PROF== Disconnected from process 325185
==PROF== Report: /home/lkb/proj-oss/whisper.cpp/my_kernel_analysis_basic.ncu-rep

```

이제 디스크에 my_kernel_analysis_basic.ncu-rep 라는 파일이 저장되었다. 확장자가 *.ncu-rep 인 파일은 프로파일링 정보를 바이너리 포멧으로 담고 있는데 이는 
nsight compute 프로그램 내에서 로드하여 확인할 수 있다

<여기에 로드한 스크린샷을  첨부할 