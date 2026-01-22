# I18N 웹사이트: hreflang vs Sitemap

둘 다 필요합니다! **hreflang 태그와 sitemap은 서로 보완 관계**입니다.

## 역할 차이

### 1. **hreflang 태그** (필수)
```html
<link rel="alternate" hreflang="ko" href="https://example.com/ko" />
<link rel="alternate" hreflang="en" href="https://example.com/en" />
<link rel="alternate" hreflang="x-default" href="https://example.com/en" />
```

**역할:**
- 같은 콘텐츠의 다른 언어 버전을 Google에게 알림
- 사용자 언어에 맞는 버전을 검색 결과에 표시
- 중복 콘텐츠 문제 방지

**선호도: ⭐⭐⭐⭐⭐ (필수)**

### 2. **Sitemap** (권장)
```xml
<url>
  <loc>https://example.com/ko</loc>
  <xhtml:link rel="alternate" hreflang="en" href="https://example.com/en"/>
  <xhtml:link rel="alternate" hreflang="ko" href="https://example.com/ko"/>
</url>
```

**역할:**
- 모든 페이지를 Google에게 알림 (발견성)
- 크롤링 우선순위 제공
- hreflang 정보도 포함 가능

**선호도: ⭐⭐⭐⭐ (강력 권장)**

## 업계 베스트 프랙티스

**결론: 둘 다 사용하는 것이 표준입니다.**

```
✅ HTML에 hreflang 태그 (각 페이지)
✅ Sitemap에 모든 언어 버전 포함
✅ 일관성 유지 (양쪽 정보가 일치해야 함)
```

## Google의 공식 권장사항

1. **작은 사이트 (<1000 페이지):** HTML hreflang만으로도 충분
2. **큰 사이트:** Sitemap에 hreflang 포함 (유지보수 편함)
3. **대규모 사이트:** HTTP 헤더 방식도 고려

## 현재 상황 분석

### Googlebot 크롤링 문제

**질문:** Googlebot이 `/en`과 `/ko` 둘 다 크롤링 가능한가?

**답변:** 아니요, 현재 코드로는 불가능합니다.

**이유:**

1. **Googlebot의 기본 Accept-Language 헤더**
   - Googlebot은 보통 `Accept-Language: en-US,en;q=0.9` 또는 빈 값을 보냅니다
   - 따라서 `acceptLanguage.startsWith("ko")`는 항상 `false`
   - 결과적으로 **항상 `/en`으로만 리다이렉트**됩니다

2. **`/ko` 페이지는 크롤링되지 않음**
   - Googlebot이 루트 `/`에 접속하면 → 항상 `/en`으로 리다이렉트
   - `/ko` 페이지는 직접 링크가 없으면 발견되지 않습니다

## 해결 방법

당신의 블로그는 **둘 다 구현**하는 것을 추천합니다:

1. **각 페이지에 hreflang 추가** (Layout.astro)
2. **Sitemap에 `/ko`, `/en` 모두 포함**
3. **루트(`/`)는 x-default로 설정**

### 구현 예시

#### Layout.astro에 추가
```astro
<head>
  <!-- 현재 페이지의 언어에 따라 동적으로 생성 -->
  <link rel="alternate" hreflang="ko" href={`https://yoursite.com${Astro.url.pathname.replace(/^\/en/, '/ko')}`} />
  <link rel="alternate" hreflang="en" href={`https://yoursite.com${Astro.url.pathname.replace(/^\/ko/, '/en')}`} />
  <link rel="alternate" hreflang="x-default" href={`https://yoursite.com${Astro.url.pathname.replace(/^\/ko/, '/en')}`} />
</head>
```

#### Sitemap 설정
Astro의 sitemap integration이 자동으로 두 언어 버전을 모두 포함하도록 설정
