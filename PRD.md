# OeFlakTrack — Product Requirements Document

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Goals & Non-Goals](#3-goals--non-goals)
- [4. Users & Stakeholders](#4-users--stakeholders)
- [5. System Architecture](#5-system-architecture)
- [6. Functional Requirements](#6-functional-requirements)
- [7. Non-Functional Requirements](#7-non-functional-requirements)
- [8. Current Implementation Status](#8-current-implementation-status)
- [9. Future Roadmap](#9-future-roadmap)
- [10. Key Algorithms Reference](#10-key-algorithms-reference)
- [한국어 PRD](#한국어-prd)

---

## 1. Overview

**OeFlakTrack** (Air Defense Artillery Flak Tracking System) is a real-time aircraft tracking platform that uses an **Extended Kalman Filter (EKF)** to estimate aircraft positions from noisy radar sensor data. The system simulates a distributed radar network and applies advanced signal-processing algorithms to deliver accurate positional estimates for use in anti-aircraft targeting and threat assessment scenarios.


---

## 2. Problem Statement

Modern radar systems suffer from inherent measurement noise caused by:
- Atmospheric interference and multipath signal reflections
- Sensor hardware limitations (range/azimuth quantization)
- High-speed target kinematics that outpace scan intervals

Raw sensor data is unreliable for precision targeting. A filtering layer is needed to **reduce noise, predict target state, and produce smooth, low-latency position estimates**.

---

## 3. Goals & Non-Goals

### Goals
- Simulate a realistic radar sensor with configurable noise parameters
- Implement an EKF that converts noisy polar measurements (range, azimuth) to filtered Cartesian state (X, Y, Vx, Vy)
- Deliver real-time tracking via a ZeroMQ publish/subscribe architecture
- Quantify performance with RMSE, latency, and throughput metrics
- Provide multi-dimensional (1D / 2D / 3D) Kalman Filter reference implementations in Python and Java

### Non-Goals
- Live RF/radar hardware integration (simulation only in current scope)
- Classified or export-controlled military system deployment
- GUI or visualization dashboard (CLI-only output in current scope)

---

## 4. Users & Stakeholders

| Role | Interest |
|------|----------|
| Aerospace/Defense Engineers | Reference implementation of EKF for radar tracking |
| Signal Processing Students | Hands-on Kalman Filter study across 1D/2D/3D |
| Robotics / Autonomous Systems Developers | Reusable sensor-fusion pattern |
| Competitive Programming / Research | Benchmark for tracking algorithm performance |

---

## 5. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                          OeFlakTrack                           │
│                                                                │
│  ┌─────────────────────┐   ZeroMQ PUB/SUB   ┌──────────────┐  │
│  │   AircraftTrajSimul  │ ──────────────────▶│ ExtfKFTracker│  │
│  │   (Radar Simulator) │    tcp://5555       │  (EKF Core)  │  │
│  │                     │    JSON payload     │              │  │
│  │  • Constant-vel     │                     │  • Predict   │  │
│  │    motion model     │                     │  • Update    │  │
│  │  • Gaussian noise   │                     │  • Jacobian  │  │
│  │    (range/azimuth)  │                     │  • Metrics   │  │
│  └─────────────────────┘                     └──────────────┘  │
│                                                                │
│  ┌────────────────────── Reference Filters ───────────────┐    │
│  │  kalman 1d.py  │  kalman 2d.py  │  kalman 3d.py        │    │
│  │  Kalman1D.java │  Kalman2D.java │  Kalman3D.java        │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Core Algorithm | Python 3, NumPy |
| Messaging | ZeroMQ (pyzmq) |
| Serialization | JSON |
| Java Reference | JavaSE-21, Apache Commons Math3 |
| Dev Environment | VS Code, Eclipse IDE |

---

## 6. Functional Requirements

### FR-1 · Radar Simulation
- Simulate aircraft trajectory using a **constant-velocity motion model** (state: X, Y, Vx, Vy)
- Inject calibrated Gaussian noise: range σ = 50 m, azimuth σ = 0.02 rad
- Publish telemetry JSON at a configurable scan interval (`dt`)
- Payload includes: `timestamp`, `noisy_range`, `noisy_angle`, `true_x`, `true_y`

### FR-2 · Extended Kalman Filter Tracking
- Subscribe to radar telemetry on `tcp://localhost:5555` with topic filter `"RADAR"`
- Maintain 4-state vector: `[X, Y, Vx, Vy]`
- Compute **Jacobian matrix** of polar measurement function for EKF linearization
- Apply **angle wrapping** (−π to +π) to prevent discontinuity errors
- Guard against singular-point division-by-zero near the radar origin
- Log filtered state every 10 seconds; display performance summary on `SIGINT`

### FR-3 · Performance Metrics
- Compute **raw sensor RMSE** (pre-filter) and **EKF RMSE** (post-filter)
- Report noise reduction percentage
- Measure per-packet processing latency (microsecond precision)
- Report maximum achievable throughput (Hz)
- Capture host hardware info (OS, processor) for reproducibility

### FR-4 · Multi-Dimensional Reference Implementations
- **1D**: Scalar Kalman filter over a single noisy series (Python + Java)
- **2D**: 4-state (X, Y, Vx, Vy) initialization matrices (Python + Java)
- **3D**: 6-state (X, Y, Z, Vx, Vy, Vz) initialization matrices (Python + Java)

---

## 7. Non-Functional Requirements

| Attribute | Target |
|-----------|--------|
| Latency | < 1 ms per EKF update |
| RMSE Reduction | > 60 % versus raw sensor |
| Scalability | Handles ≥ 100 messages/second |
| Portability | Runs on macOS, Linux, Windows (Python 3.10+) |
| Maintainability | Each module ≤ 200 lines; clear docstrings |

---

## 8. Current Implementation Status

| Component | Status |
|-----------|--------|
| `AircraftTrajSimul.py` — Radar simulator | ✅ Complete |
| `ExtfKFTracker.py` — EKF tracker + profiler | ✅ Complete |
| `kalman 1d.py` — 1D Python filter | ✅ Complete |
| `kalman 2d.py` — 2D Python init | ✅ Complete |
| `kalman 3d.py` — 3D Python init | ✅ Complete |
| `Kalman1D.java` — 1D Java filter | ✅ Complete |
| `Kalman2D.java` — 2D Java matrices | ⚠️ Init only (no filter loop) |
| `Kalman3D.java` — 3D Java matrices | ⚠️ Init only (no filter loop) |

---

## 9. Future Roadmap

The following improvements are planned for upcoming commits, ordered by priority:

### v1.1 — Complete Java Implementations
- Add predict/update cycle to `Kalman2D.java` and `Kalman3D.java`
- Mirror the Python EKF logic in Java using Apache Commons Math3 `RealMatrix`
- Add JUnit 5 unit tests for each Java filter class

### v1.2 — 3D Altitude Tracking
- Extend the Python EKF state vector to `[X, Y, Z, Vx, Vy, Vz]`
- Update measurement function to handle spherical coordinates (range, azimuth, elevation)
- Re-derive the 6×3 Jacobian for the extended state

### v1.3 — Maneuvering Target Model
- Replace constant-velocity with **Singer acceleration model** or **Interacting Multiple Model (IMM)**
- Add process noise tuning parameters for target maneuver intensity
- Validate against synthetic evasive maneuver trajectories

### v1.4 — Multi-Target Tracking
- Implement **Global Nearest Neighbor (GNN)** data association
- Support simultaneous tracking of N independent aircraft
- Introduce track initiation and termination logic

### v1.5 — Adaptive Noise Estimation
- Implement **Adaptive Kalman Filter** that auto-tunes Q and R matrices using innovation covariance monitoring
- Remove need for manual noise parameter calibration

### v1.6 — Visualization & Dashboard
- Add `matplotlib` trajectory plots comparing raw measurements vs. EKF estimates
- Export telemetry to CSV/HDF5 for post-run analysis
- Optional: Real-time `pygame` or `OpenGL` radar scope display

### v2.0 — Multi-Sensor Fusion
- Fuse data from multiple simulated radar sites using **Federated Kalman Filter**
- Add simulated infrared (IR) sensor with different noise characteristics
- Implement **Track-to-Track Fusion** architecture

---

## 10. Key Algorithms Reference

### EKF Equations

**Prediction:**
```
x̂⁻ = F · x̂
P⁻  = F · P · Fᵀ + Q
```

**Update:**
```
ẑ   = h(x̂⁻)            # nonlinear measurement prediction
y   = z − ẑ             # innovation (residual)
Hj  = ∂h/∂x|x̂⁻         # Jacobian of measurement function
S   = Hj · P⁻ · Hjᵀ + R
K   = P⁻ · Hjᵀ · S⁻¹   # Kalman gain
x̂   = x̂⁻ + K · y
P   = (I − K · Hj) · P⁻
```

**Measurement function (polar):**
```
h(x) = [ sqrt(x² + y²),  atan2(y, x) ]
```

**Jacobian:**
```
Hj = [ x/r,   y/r,  0, 0 ]
     [ -y/c,  x/c,  0, 0 ]

where r = sqrt(x²+y²),  c = x²+y²
```

*This document reflects all current and planned features of the project and will be updated with each release.*

---

---

# 한국어 PRD

## 목차

- [1. 개요](#1-개요)
- [2. 문제 정의](#2-문제-정의)
- [3. 목표 및 비목표](#3-목표-및-비목표)
- [4. 사용자 및 이해관계자](#4-사용자-및-이해관계자)
- [5. 시스템 아키텍처](#5-시스템-아키텍처)
- [6. 기능 요구사항](#6-기능-요구사항)
- [7. 비기능 요구사항](#7-비기능-요구사항)
- [8. 현재 구현 상태](#8-현재-구현-상태)
- [9. 향후 개선 로드맵](#9-향후-개선-로드맵)
- [10. 핵심 알고리즘 참조](#10-핵심-알고리즘-참조)

---

## 1. 개요

**OeFlakTrack** (외 대공포 항공기 추적 시스템)은 **확장 칼만 필터(EKF)**를 사용하여 노이즈가 포함된 레이더 센서 데이터로부터 항공기 위치를 실시간으로 추정하는 플랫폼입니다. 분산 레이더 네트워크를 시뮬레이션하고 고급 신호 처리 알고리즘을 적용하여 대공 표적 식별 및 위협 평가 시나리오에 활용 가능한 정확한 위치 추정값을 제공합니다.

---

## 2. 문제 정의

현대 레이더 시스템은 아래와 같은 원인으로 고유한 측정 노이즈가 발생합니다:
- 대기 간섭 및 다중 경로 신호 반사
- 센서 하드웨어 한계(거리/방위각 양자화 오차)
- 스캔 주기를 초과하는 고속 표적의 기동성

가공되지 않은 센서 데이터는 정밀 표적화에 신뢰할 수 없습니다. **노이즈 감소, 표적 상태 예측, 부드럽고 저지연의 위치 추정값**을 생성하는 필터링 계층이 필요합니다.

---

## 3. 목표 및 비목표

### 목표
- 설정 가능한 노이즈 파라미터를 가진 현실적인 레이더 센서 시뮬레이션
- 극좌표 측정값(거리, 방위각)을 필터링된 직교 상태(X, Y, Vx, Vy)로 변환하는 EKF 구현
- ZeroMQ 발행/구독 아키텍처를 통한 실시간 추적 제공
- RMSE, 지연 시간, 처리량 메트릭으로 성능 정량화
- Python 및 Java로 다차원(1D/2D/3D) 칼만 필터 참조 구현 제공

### 비목표
- 실제 RF/레이더 하드웨어 연동 (현재 범위는 시뮬레이션 전용)
- 기밀 또는 수출 통제 군사 시스템 배포
- GUI 또는 시각화 대시보드 (현재 범위는 CLI 출력 전용)

---

## 4. 사용자 및 이해관계자

| 역할 | 관심사 |
|------|--------|
| 항공우주/국방 엔지니어 | 레이더 추적용 EKF 참조 구현 |
| 신호 처리 학생 | 1D/2D/3D 칼만 필터 실습 학습 |
| 로보틱스/자율 시스템 개발자 | 재사용 가능한 센서 퓨전 패턴 |
| 연구자 / 경쟁 프로그래머 | 추적 알고리즘 성능 벤치마크 |

---

## 5. 시스템 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                          OeFlakTrack                           │
│                                                                │
│  ┌─────────────────────┐   ZeroMQ PUB/SUB   ┌──────────────┐  │
│  │   AircraftTrajSimul  │ ──────────────────▶│ ExtfKFTracker│  │
│  │   (레이더 시뮬레이터) │    tcp://5555       │  (EKF 코어)  │  │
│  │                     │    JSON 페이로드    │              │  │
│  │  • 등속 운동 모델   │                     │  • 예측      │  │
│  │  • 가우시안 노이즈  │                     │  • 업데이트  │  │
│  │    (거리/방위각)    │                     │  • 야코비안  │  │
│  │                     │                     │  • 메트릭   │  │
│  └─────────────────────┘                     └──────────────┘  │
│                                                                │
│  ┌────────────────── 참조 필터 구현 ──────────────────────┐    │
│  │  kalman 1d.py  │  kalman 2d.py  │  kalman 3d.py        │    │
│  │  Kalman1D.java │  Kalman2D.java │  Kalman3D.java        │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

### 기술 스택

| 계층 | 기술 |
|------|------|
| 핵심 알고리즘 | Python 3, NumPy |
| 메시징 | ZeroMQ (pyzmq) |
| 직렬화 | JSON |
| Java 참조 구현 | JavaSE-21, Apache Commons Math3 |
| 개발 환경 | VS Code, Eclipse IDE |

---

## 6. 기능 요구사항

### FR-1 · 레이더 시뮬레이션
- **등속 운동 모델**을 사용한 항공기 궤적 시뮬레이션 (상태: X, Y, Vx, Vy)
- 보정된 가우시안 노이즈 주입: 거리 σ = 50 m, 방위각 σ = 0.02 rad
- 설정 가능한 스캔 간격(`dt`)으로 텔레메트리 JSON 발행
- 페이로드 포함 항목: `timestamp`, `noisy_range`, `noisy_angle`, `true_x`, `true_y`

### FR-2 · 확장 칼만 필터 추적
- 토픽 필터 `"RADAR"`로 `tcp://localhost:5555`의 레이더 텔레메트리 구독
- 4-상태 벡터 유지: `[X, Y, Vx, Vy]`
- EKF 선형화를 위한 극좌표 측정 함수의 **야코비안 행렬** 계산
- 불연속 오류 방지를 위한 **각도 랩핑** (−π ~ +π) 적용
- 레이더 원점 근처에서 특이점 0 나눗셈 방지
- 10초마다 필터링된 상태 로그 출력; `SIGINT` 시 성능 요약 표시

### FR-3 · 성능 메트릭
- **원시 센서 RMSE** (필터 전) 및 **EKF RMSE** (필터 후) 계산
- 노이즈 감소율 보고
- 패킷당 처리 지연 시간 측정 (마이크로초 정밀도)
- 최대 처리량 보고 (Hz)
- 재현성을 위한 호스트 하드웨어 정보 수집 (OS, 프로세서)

### FR-4 · 다차원 참조 구현
- **1D**: 단일 노이즈 시계열에 대한 스칼라 칼만 필터 (Python + Java)
- **2D**: 4-상태 (X, Y, Vx, Vy) 초기화 행렬 (Python + Java)
- **3D**: 6-상태 (X, Y, Z, Vx, Vy, Vz) 초기화 행렬 (Python + Java)

---

## 7. 비기능 요구사항

| 속성 | 목표 |
|------|------|
| 지연 시간 | EKF 업데이트당 < 1 ms |
| RMSE 감소율 | 원시 센서 대비 > 60 % |
| 확장성 | 초당 ≥ 100 메시지 처리 |
| 이식성 | macOS, Linux, Windows (Python 3.10+) 실행 가능 |
| 유지보수성 | 모듈당 ≤ 200 줄; 명확한 독스트링 |

---

## 8. 현재 구현 상태

| 컴포넌트 | 상태 |
|---------|------|
| `AircraftTrajSimul.py` — 레이더 시뮬레이터 | ✅ 완료 |
| `ExtfKFTracker.py` — EKF 추적기 + 프로파일러 | ✅ 완료 |
| `kalman 1d.py` — 1D Python 필터 | ✅ 완료 |
| `kalman 2d.py` — 2D Python 초기화 | ✅ 완료 |
| `kalman 3d.py` — 3D Python 초기화 | ✅ 완료 |
| `Kalman1D.java` — 1D Java 필터 | ✅ 완료 |
| `Kalman2D.java` — 2D Java 행렬 | ⚠️ 초기화만 (필터 루프 없음) |
| `Kalman3D.java` — 3D Java 행렬 | ⚠️ 초기화만 (필터 루프 없음) |

---

## 9. 향후 개선 로드맵

다음 커밋에서 우선순위 순으로 계획된 개선 사항입니다:

### v1.1 — Java 구현 완성
- `Kalman2D.java` 및 `Kalman3D.java`에 예측/업데이트 사이클 추가
- Apache Commons Math3 `RealMatrix`를 사용하여 Python EKF 로직을 Java로 이식
- 각 Java 필터 클래스에 JUnit 5 단위 테스트 추가

### v1.2 — 3D 고도 추적
- Python EKF 상태 벡터를 `[X, Y, Z, Vx, Vy, Vz]`로 확장
- 구면 좌표계(거리, 방위각, 앙각) 처리를 위한 측정 함수 업데이트
- 확장 상태에 대한 6×3 야코비안 재도출

### v1.3 — 기동 표적 모델
- 등속도 모델을 **Singer 가속 모델** 또는 **상호작용 다중 모델(IMM)**로 교체
- 표적 기동 강도에 대한 프로세스 노이즈 튜닝 파라미터 추가
- 합성 회피 기동 궤적으로 검증

### v1.4 — 다중 표적 추적
- **전역 최근접 이웃(GNN)** 데이터 연관 구현
- N대의 독립 항공기 동시 추적 지원
- 트랙 시작 및 종료 논리 도입

### v1.5 — 적응형 노이즈 추정
- 혁신 공분산 모니터링을 사용하여 Q 및 R 행렬을 자동 튜닝하는 **적응형 칼만 필터** 구현
- 수동 노이즈 파라미터 보정 필요성 제거

### v1.6 — 시각화 및 대시보드
- 원시 측정값과 EKF 추정값을 비교하는 `matplotlib` 궤적 플롯 추가
- 사후 분석을 위한 CSV/HDF5 텔레메트리 내보내기
- 선택 사항: 실시간 `pygame` 또는 `OpenGL` 레이더 스코프 디스플레이

### v2.0 — 다중 센서 퓨전
- **연합 칼만 필터**를 사용하여 복수 시뮬레이션 레이더 기지의 데이터 융합
- 다른 노이즈 특성을 가진 시뮬레이션 적외선(IR) 센서 추가
- **트랙 간 퓨전** 아키텍처 구현

---

## 10. 핵심 알고리즘 참조

### EKF 방정식

**예측 단계:**
```
x̂⁻ = F · x̂
P⁻  = F · P · Fᵀ + Q
```

**업데이트 단계:**
```
ẑ   = h(x̂⁻)            # 비선형 측정 예측
y   = z − ẑ             # 혁신(잔차)
Hj  = ∂h/∂x|x̂⁻         # 측정 함수의 야코비안
S   = Hj · P⁻ · Hjᵀ + R
K   = P⁻ · Hjᵀ · S⁻¹   # 칼만 게인
x̂   = x̂⁻ + K · y
P   = (I − K · Hj) · P⁻
```

**측정 함수 (극좌표):**
```
h(x) = [ sqrt(x² + y²),  atan2(y, x) ]
```

**야코비안:**
```
Hj = [ x/r,   y/r,  0, 0 ]
     [ -y/c,  x/c,  0, 0 ]

여기서 r = sqrt(x²+y²),  c = x²+y²
```

---

*이 문서는 프로젝트의 모든 현재 및 계획된 기능을 반영하며, 각 릴리즈마다 업데이트됩니다.*
