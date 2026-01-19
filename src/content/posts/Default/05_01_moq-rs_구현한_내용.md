---
title: 05_01_moq-rs_구현한_내용
published: 0001-01-01
description: ""
tags: []
category: Default
draft: true
---

## 이 글의 목적

## 이 글에 사용되는 기반 지식들

- 웹 트랜스포트 프로토콜
- MoQ(Media over QUIC) 또는 MoQT(Media over QUIC Transport) 프로토콜
- ffmpeg로 실시간 캡쳐하기 

## ffmpeg로 실시간 캡처하는 구체적인 방법

비디오아 오디오 실시간 캡처는 다음 ffmpeg 명령어로 수행한다
이 명령어가 무슨 의미인지는 나중에 설명하겠다

```
ffmpeg -loglevel info -hide_banner `
-init_hw_device d3d11va:,vendor_id=0x8086 `
-itsoffset -0.7 -f dshow -i audio="CABLE Output(VB-Audio Virtual Cable)" `
-filter_complex "ddagrab=1:framerate=60,hwmap=derive_device=qsv,format=qsv" `
-af "aresample=async=1000" `
-c:v hevc_qsv `
-c:a aac -b:a 128k `
-preset veryfast `
-global_quality 25 `
-b:v 4500k -maxrate 5000k -bufsize 13000k `
-g 1 -keyint_min 1 `
-shortest `
-f mp4 -movflags empty_moov+default_base_moof+frag_keyframe `
- | cargo run --bin hang -- publish --url "http://localhost:4443/anon" --name "screen"
```

-itsoffset -0.7: You are delaying the audio. This is fine if you've noticed a sync issue, but make sure the "Broken pipe" isn't masking a synchronization crash (though in this specific log, the Cargo.toml is definitely the main culprit).
-g 1: You have set the GOP size to 1. This means every single frame is a Keyframe (All-Intra). This will result in very high quality but will use much more bandwidth than 4500k usually allows. If you see stuttering, increase this to -g 60 or -g 120.


## ffmpeg에서 비디오를 캡처하는 방법

- 비디오 캡처는 ffmpeg의 ddagrab 필터로 수행된다 

이 필터는 Windows 8부터 제공되는 windows API인 Desktop Duplication API를 사용하여 이론상으로 가장 빠른 캡쳐 속도를 자랑한다 (이 내용이 맞는지 체크할 것)

상세 정보는 다음을 참조할 것 : https://trac.ffmpeg.org/wiki/Capture/Desktop#UseWindows8DesktopDuplicationAPI

ffmpeg 명령어에 다음 옵션을 추가한다

-filter_complex "ddagrab=1:framerate=60,hwmap=derive_device=qsv,format=qsv;[0:a]aresample=async=1000[audio]" `

## ffmpeg에서 오디오를 캡처하는 방법

- 오디오 캡처도 ddagrab 필터로 수행하면 좋겠지만 ddagrab 필터는 오직 비디오 캡처만을 지원한다 
- 그래서 ffmpeg의 `dshow` 필터를 사용해여 캡쳐한다. 

ffmpeg 명령어에 다음 옵션을 추가한다

-f dshow -rtbufsize 100M -audio_buffer_size 0 -i audio="마이크(JOYTRON HD20)" `

- 방법 1 : dshow(DirectShow) 필터를 사용한다 (마이크 캡처)
```
-f dshow -rtbufsize 100M -audio_buffer_size 0 -i audio="마이크(JOYTRON HD20)" `
```

(틀린내용) 위와 같이 마이크 캡처를 하는 이유는 오디오 스트림을 직접 output하는것이 불가능하기 때문이다

그러므로 VB-AUDIO라는 프로그램을 사용하여 오디오를 캡처한다

참고 : https://vb-audio.com/Cable/

한번 설치 이후에는 별도의 프로그램 실행이 필요없다

설치후 아래와 같이 사용한다

```
-f dshow -i audio="CABLE Output(VB-Audio Virtual Cable)"
```

## dshow(direct show) 필터에서 오디오 딜레이 추가하기

dshow에 `-itsoffset` 옵션을 사용한다. 단위는 초(seconds)이다

```
-itsoffset -2.0 -f dshow -i audio="CABLE Output(VB-Audio Virtual Cable)"
```

## 결과물

AWS EC2 서버를 작동하는 서버를 만들었다. 이 서버는 rust로 작성된 moq-rs를 기반으로 구동된다.
상세 코드는 `G:\my\moq-related\moq-rs-ec2`에서 확인할 수 있다

이 서버와의 연결은 웹브라우저에서 이루어진다. 상세는 `G:\my\moq-related\moq-rs-ec2\js` 를 참조하라.

## 결과물의 프레임레이트

만일 키프레임을 1ms단위로 조정한다면 60프레임 화면을 송출할 수 있다


## 실시간 캡처의 문제점

- 스트리밍 화질이 낮다. 
	- 키프레임 간격을 줄이면서부터 화질이 낮아진 이유는 -> 화질 높일것
	- 문제의 원인 : 키프레임 간격이 너무 짦음
	- 모든 프레임을 키프레임으로 설정해놨음 (이렇게 설정하니 delay가 줄어듬)

- 뚝뚝 끊긴다. 끊기는 이유는 GPU 사용량이 너무 많은것이 원인으로 의심된다
	- 웹 클라이언트에서 비디오를 디코딩할 때 GPU 사용량이 약 85% 정도로 치솟는다
	- 이 상황에서 반디캠 녹화를 실행하면 GPU 사용량이 99%로 치솟는다. 이 경우 유튜브도 끊기고 웹 클라이언트도 끊기고 다 끊긴다.
	- 이 상황에서는 ffmpeg에 브로큰 파이프 에러가 발생한다
		- `Error submitting a packet to the muxer: Broken pipeA`
		- 즉 브로큰 파이프 에러는 GPU 사용량이 너무 많을 때 발생한다. GPU 사용량이 적은 상황에서는 FFMPEG 전송이 중단없이 1시간이 넘어도 정상 수행된다.

- ffmpeg 캡쳐시 `GPU 사용률이 3초 95% → 10초 유휴 반복`된다 
	- 실시간 캡처이므로 gpu 사용량이 모든 순간에 균등하게 유지되어야 정상이다. 그런데 `사용률 피크 이후 유휴시간 ->	사용률 피크 이후 유휴시간` 의 패턴이 무한 반복된다

---

## 왜 이 결과물이 딜레이가 충분히 느린가 ? 

여기에는 두가지 핵심이 있다. 
첫번째. ffmpeg에서 제공하는 코덱인 


## 결과물의 결함

이 x는 오디오가 들리지 않는다


## 이 결함을 해결하는 방법

ffmpeg의 옵션중 ㅇㅇㅇ이 있는데 이 옵션의 한계이다. 이 옵션은 비디오와 오디오를 동시에 캡쳐하는 기능이 없다.

그래서 ㅇㅇㅇ 가 아닌 ㅌㅌㅌ를 사용하여 오디오를 캡쳐해야 한다. 이 때 오디오 딜레이가 발생하여 싱크가 맞지 않는다. 

이 싱크를 맞추기 위해 다음 옵션을 추가한다 

< 여기에 옵션을 기재할 것 >


## ffmpeg 캡처관련 참고할 웹사이트

- ddagrab 캡처관련 : [how can I capture audio and also stream that ?](https://www.reddit.com/r/ffmpeg/comments/yncqy8/i_have_discovered_how_to_efficiently_capture/)
- ddagrab 오디오 싱크 관련 정보 : [ddagrab with dshow audio - out of sync](https://www.reddit.com/r/ffmpeg/comments/1lw891y/ddagrab_with_dshow_audio_out_of_sync/)