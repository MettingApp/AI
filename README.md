# Mote AI — Meeting Transcription & Summarization

회의 지원 앱 **Mote**의 한국어 음성 인식, 화자 분리, 회의 요약을 위한 AI 처리 모듈입니다. 회의 녹음에서 화자별 발화를 추출하고 한국어 전사문을 만든 뒤, 주제별 요약·결론·추가 조사 키워드·다음 회의 일정으로 정리하는 흐름을 구현했습니다.

**AI 담당: Gyutak Hahn · 한규탁**  
음성 데이터 구축, Whisper 학습·평가, 화자 분리, 모델 경량화, 요약 호출 로직을 담당하고 백엔드 팀원과 AI 입출력·앱 연동을 협업했습니다.

[Whisper Training](https://github.com/HahnGyuTak/openai-whisper-finetuning) · [Models & Datasets](https://huggingface.co/imTak) · [Team Backend](https://github.com/MettingApp/backend) · [iOS App](https://github.com/MettingApp/iOS) · [Presentation](https://github.com/user-attachments/files/19148912/Mote.pdf)

## My Contribution

| 영역 | 직접 수행한 내용 |
|---|---|
| 한국어 STT 학습 | Whisper large-v3-turbo를 한국어 데이터로 파인튜닝하고, 개발·경제 분야 음성–텍스트 데이터를 구성해 추가 학습했습니다. WER 평가와 체크포인트 선택을 연결하고 모델·데이터셋을 공개했습니다. |
| 음성 데이터 구축 | 영상 음성과 한국어 자막을 수집해 자막 구간별 음성–정답 쌍을 구성했습니다. 최종 16 kHz 변환, 1초 이하 샘플 제외, 학습·평가 데이터 분할과 Hub 업로드를 수행했습니다. |
| 화자 분리·전사 연결 | `simple_diarizer`의 x-vector 임베딩과 spectral clustering을 활용해 회의 참여자의 발화를 구분했습니다. 분리한 구간을 WAV로 저장하고 STT 결과에 A–H 화자 라벨을 붙이는 처리 함수를 구현했습니다. |
| 추론 적용 | 학습 모델을 CTranslate2 형식으로 변환·양자화하고 faster-whisper로 CPU 추론을 구성했습니다. 발표자료 기준 모델 파일 용량은 **3.2 GB → 820 MB**입니다. |
| 회의 요약 | 화자별 전사문을 입력받아 주제별 요약·결론·검색 키워드·다음 회의 일정을 Markdown으로 생성하는 LLM 호출과 프롬프트를 작성했습니다. |
| 팀 서비스 연동 | 백엔드 담당 팀원과 녹음 파일·화자 수·회의 분야 등의 입력과 요약 결과 전달 방식을 맞추고 AI 기능의 앱 연동을 수행했습니다. 아래 Spring API는 팀 백엔드 구현이며 AI 담당 개인 기여와 구분합니다. |

학습 설정·WER 기록·데이터셋 규모는 [Whisper 학습 저장소 README](https://github.com/HahnGyuTak/openai-whisper-finetuning#readme)에 정리했습니다. 이 저장소는 학습 결과를 회의 처리에 연결하는 코드와 입력 인터페이스를 중심으로 설명합니다.

## Processing Flow

[`wav2md.py`](./wav2md.py)의 처리 함수는 다음 순서로 연결됩니다.

```text
회의 녹음 WAV
   │
   ├─ 로컬 파일 입력
   └─ S3 오브젝트 다운로드: download_file_from_s3()
   ↓
speaker_diarization()
   x-vector + spectral clustering · 참여 화자 수 지정
   2초 이상 발화 구간을 화자별 WAV로 저장
   ↓
stt_transcription()
   faster-whisper · 한국어 전사 · A–H 화자 라벨
   output/RESULT.txt
   ↓
llm_summarization()
   전사문 → 주제별 요약 / 결론 / 검색어 / 다음 회의 일정
   Markdown 문자열
```

| 함수 | 입력 | 처리 결과 |
|---|---|---|
| `speaker_diarization()` | diarizer, WAV 경로, 화자 수, 출력 폴더 | 발화별 WAV 파일 경로 목록 |
| `stt_transcription()` | STT 모델, 발화 WAV 목록, 출력 폴더 | `RESULT.txt`에 `[구간 내 시간] 화자: 전사문` 기록 |
| `llm_summarization()` | OpenAI client, 전사문 경로 | 요약 Markdown 문자열 |
| `download_file_from_s3()` | bucket, object key, 로컬 경로 | S3 파일 다운로드 |
| `handler(event)` | S3 입력 이벤트 | 아래 공개 코드 상태 참고 |

### Current Implementation Status

현재 공개 `main`은 AI 핵심 함수와 S3 연동 실험 코드를 담은 스냅샷입니다.

- `handler(event)`는 S3 다운로드 직후 `return`하므로, 기본 실행으로는 뒤의 화자 분리·전사·요약에 도달하지 않습니다. 아래 **Local Run** 예제는 각 함수를 직접 순서대로 호출합니다.
- 함수 하단의 `result` 응답 딕셔너리는 작성돼 있지만 반환되지 않습니다. `organization_field`도 읽기만 하며 모델 선택에는 사용하지 않습니다.
- 이 저장소에는 FastAPI 앱이나 HTTP 라우트가 없습니다. HTTP 회의 API는 별도의 [팀 백엔드](https://github.com/MettingApp/backend)에서 제공합니다.
- 원본 요약 함수의 모델은 `gpt-3.5-turbo`입니다. 이 모델은 **2026-10-23에 API 지원이 종료**됐으므로 요약을 다시 실행할 때 모델 설정을 변경해야 합니다. [OpenAI 공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-3.5-turbo)

## Local Run

### 1. Environment

프로젝트의 기존 환경은 Python 3.10입니다. 화자 분리에 필요한 `simple_diarizer`가 있는 서버용 의존성 목록을 사용하고, 소스에서 import하는 `boto3`·`soundfile`을 추가합니다.

```bash
git clone https://github.com/MettingApp/AI.git
cd AI

conda create -n mote-ai python=3.10 -y
conda activate mote-ai
conda install -c conda-forge ffmpeg -y

python -m pip install -r requirements-server.txt
python -m pip install boto3 soundfile tqdm
```

첫 실행에는 diarizer와 Whisper 모델 다운로드가 필요합니다. `requirements-server.txt`는 당시의 버전 구성이며 모든 의존성이 고정된 lockfile은 아닙니다. PyTorch와 torchaudio는 서로 맞는 버전을 사용합니다.

### 2. Input & Credentials

녹음을 mono·16 kHz WAV로 준비합니다.

```bash
ffmpeg -i meeting.m4a -ac 1 -ar 16000 meeting.wav
```

`wav2md.py`는 모듈 import 시 `API_KEY.txt`를 읽습니다. **STT까지만 실행할 경우** 빈 파일을 만들면 아래 예제에서 요약 API를 호출하지 않고 진행할 수 있습니다.

```bash
touch API_KEY.txt
```

**요약까지 실행할 경우** 터미널 환경에 본인의 `OPENAI_API_KEY`를 설정한 뒤 다음 명령으로 파일에 저장합니다. `API_KEY.txt`는 기존 `.gitignore`에 포함돼 있습니다.

```bash
python - <<'PY'
import os
from pathlib import Path
Path("API_KEY.txt").write_text(os.environ["OPENAI_API_KEY"].strip(), encoding="utf-8")
PY
```

이어서 [`llm_summarization()`](./wav2md.py)의 `model="gpt-3.5-turbo"`를 현재 계정에서 사용할 수 있는 **Chat Completions 지원 모델**로 변경합니다. 모델별 지원 파라미터도 확인합니다. 아래 예제의 `RUN_SUMMARY=True`는 이 설정 이후 사용합니다. [OpenAI Chat Completions 문서](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

### 3. Run the Processing Functions

다음 코드를 저장소 루트의 `run_local.py`로 저장합니다. 기본값은 **화자 분리·STT까지만 실행**합니다. 요약을 켜면 생성한 전사문을 OpenAI API로 전송합니다.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from faster_whisper import WhisperModel
from openai import OpenAI
from simple_diarizer.diarizer import Diarizer
from wav2md import speaker_diarization, stt_transcription, llm_summarization

WAV_FILE = "meeting.wav"
NUM_SPEAKERS = 3
RUN_SUMMARY = False  # 모델·인증 설정을 마친 뒤 True로 변경

if not 1 <= NUM_SPEAKERS <= 8:
    raise ValueError("화자 수는 현재 A–H 라벨 범위인 1–8명으로 지정하세요.")

output = Path("output/local-demo")
output.mkdir(parents=True, exist_ok=True)

diarizer = Diarizer(embed_model="xvec", cluster_method="sc")
stt_model = WhisperModel("imTak/faster-whisper_Korean_L3turbo", device="cpu")

# 실행마다 새 임시 폴더를 사용해 이전 회의의 발화 파일이 섞이지 않게 합니다.
with TemporaryDirectory(prefix="segments-", dir=output) as temp:
    segments = speaker_diarization(
        diarizer, WAV_FILE, NUM_SPEAKERS, str(Path(temp) / "speakers")
    )
    # 원본 함수의 문자열 정렬 대신 파일명 시작 시각을 숫자로 정렬합니다.
    segments = sorted(segments, key=lambda p: int(Path(p).name.split("_", 1)[0]))
    if not segments:
        raise RuntimeError("2초 이상 발화 구간이 없습니다. 입력 음성과 화자 수를 확인하세요.")
    stt_transcription(stt_model, segments, temp)
    transcript = output / "RESULT.txt"
    shutil.copyfile(Path(temp) / "RESULT.txt", transcript)

print(f"Transcript: {transcript}")

if RUN_SUMMARY:
    key = Path("API_KEY.txt").read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("요약에는 API_KEY.txt 설정이 필요합니다.")
    summary = llm_summarization(OpenAI(api_key=key), str(transcript))
    summary_path = output / "SUMMARY.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Summary: {summary_path}")
```

```bash
python run_local.py
```

```text
output/local-demo/
├── RESULT.txt       # 화자별 전사문
└── SUMMARY.md       # RUN_SUMMARY=True일 때 생성
```

현재 전사문의 시간은 **각 발화 클립 내부 기준**입니다. 전체 회의 시각을 복원한 타임스탬프나 화자 이름 자동 식별 결과로 해석하지 않습니다. 2초 미만 발화는 원본 화자 분리 함수에서 제외됩니다.

## AI Event Interface

S3 연동 입력은 [`handler(event)`](./wav2md.py)에서 사용하는 다음 구조입니다. `fileName`은 공개 다운로드 URL이 아니라 **S3 object key**입니다.

```json
{
  "fileName": "recordings/meeting.wav",
  "num_speakers": 3,
  "organization_field": "Culture"
}
```

| 필드 | 형식 | 코드에서의 사용 |
|---|---|---|
| `fileName` | string | 고정 bucket `metting-bucket`에서 내려받을 object key |
| `num_speakers` | integer | 조기 반환 이후 AI 처리부에서 화자 수로 전달 |
| `organization_field` | string | 조기 반환 이후 읽지만 현재 모델 분기에 사용하지 않음 |

S3 adapter를 다시 연결할 때는 AWS 인증·bucket·object key를 환경에 맞추고 로컬 폴더를 준비해야 합니다. 현재 저장 경로는 상대 경로 `tmp/input_audio.wav`입니다. 주석의 Lambda `/tmp`와 다르며, 함수 시그니처도 `handler(event)` 한 인자이므로 표준 Lambda 핸들러로 바로 배포된다고 가정하지 않습니다.

함수 하단에 작성된 응답 형식은 아래와 같습니다. **현재 공개 handler가 반환하는 응답은 아닙니다.**

```json
{
  "statusCode": 200,
  "headers": {"Content-Type": "application/json"},
  "body": "{\"result\": \"회의 요약 Markdown\"}"
}
```

## Team Backend API

아래 경로는 AI 모듈 자체의 라우트가 아니라 [MettingApp/backend](https://github.com/MettingApp/backend)의 Spring Boot API입니다. `{id}`는 팀 ID이며, 요청에는 `Authorization: Bearer <JWT>`가 필요합니다.

| Method | Path | 역할 |
|---|---|---|
| POST | `/api/v1/meeting/{id}` | 녹음 파일을 포함한 회의 생성 |
| POST | `/api/v1/meeting/no-record/{id}` | 녹음 없이 회의 생성 |
| GET | `/api/v1/meeting/{id}` | 회의 목록 조회 |
| GET | `/api/v1/meeting/detail/{id}?meetingId=...` | 회의 상세 조회 |

### Meeting Upload Example

녹음 포함 생성 요청은 multipart의 **`req` JSON과 `file` 음성 파일**로 구성합니다. `date`는 `YYYY-MM-DD`, `members`는 서버에 존재하는 참여자 계정명 목록입니다.

```bash
# MOTE_API_BASE, TEAM_ID, MOTE_ACCESS_TOKEN을 실행 환경에 설정한 뒤 사용합니다.
curl -X POST "$MOTE_API_BASE/api/v1/meeting/$TEAM_ID" \
  -H "Authorization: Bearer $MOTE_ACCESS_TOKEN" \
  -F 'req={"date":"2024-11-24","extraContent":"개발 회의","title":"주간 회의","fileName":"meeting.wav","category":"Culture","members":["member1","member2"]};type=application/json' \
  -F 'file=@meeting.wav;type=audio/wav'
```

생성 API는 `meeting created` 성공 메시지를 반환합니다. 현재 공개 `MeetingService.createMeeting()`에서 확인되는 동작은 S3 업로드와 녹음·회의 DB 저장입니다. AI handler 호출이나 `summary` 저장 코드는 이 경로에 포함돼 있지 않습니다. 따라서 위 업로드 요청이 곧바로 AI 요약을 반환한다고 설명하지 않습니다.

회의 상세 응답에는 `title`, `date`, `extraContent`, `summary`, `recorder`(`fileName`, `recordFile`)가 있습니다. AI 결과를 다시 연결하려면 업로드한 S3 object key 전달, AI 응답 반환, 백엔드의 요약 저장·조회 경로를 함께 맞춰야 합니다.

근거: [MeetingController](https://github.com/MettingApp/backend/blob/main/src/main/java/server/meeting/domain/meeting/controller/MeetingController.java) · [MeetingCreateReq](https://github.com/MettingApp/backend/blob/main/src/main/java/server/meeting/domain/meeting/dto/MeetingCreateReq.java) · [MeetingService](https://github.com/MettingApp/backend/blob/main/src/main/java/server/meeting/domain/meeting/service/MeetingService.java)

## Models & Training Data

| 용도 | 공개 결과물 |
|---|---|
| 현재 처리 코드의 STT | [faster-whisper_Korean_L3turbo](https://huggingface.co/imTak/faster-whisper_Korean_L3turbo) |
| 한국어 Transformers 모델 | [whisper_large_v3_turbo_Korean2](https://huggingface.co/imTak/whisper_large_v3_turbo_Korean2) |
| 개발 분야 | [Transformers 모델](https://huggingface.co/imTak/whisper_large_v3_turbo_korean_Develop) · [faster-whisper 모델](https://huggingface.co/imTak/faster-whisper_large_v3_turbo_korean_Develop) · [데이터셋](https://huggingface.co/datasets/imTak/korean-audio-text-develop) |
| 경제 분야 | [Transformers 모델](https://huggingface.co/imTak/whisper_large_v3_turbo_korean_Economy) · [faster-whisper 모델](https://huggingface.co/imTak/faster-whisper_large_v3_turbo_korean_Economy) · [데이터셋](https://huggingface.co/datasets/imTak/korean-audio-text-economy) |

[`make_hug_ds.py`](./make_hug_ds.py)와 [`make_hf_ds_economy.py`](./make_hf_ds_economy.py)는 수집한 음성·한국어 자막을 분할해 `name: Audio(16000)`·`text` 필드로 구성하고 1초 이하 샘플을 제외합니다. 80:20 분할과 Hugging Face 공개를 수행하며, 직접 만든 도메인 모델의 학습·WER 평가는 [별도 학습 저장소](https://github.com/HahnGyuTak/openai-whisper-finetuning)에서 확인할 수 있습니다.

## Repository Guide

| 파일 | 내용 |
|---|---|
| [`wav2md.py`](./wav2md.py) | 화자 분리·STT·요약 함수와 S3 handler |
| [`make_hug_ds.py`](./make_hug_ds.py) | 영상별 음성·자막 처리, 개발 데이터 구성·공개 |
| [`make_hf_ds_economy.py`](./make_hf_ds_economy.py) | 경제 분야 수집·분할·Dataset 공개 |
| [`requirements-server.txt`](./requirements-server.txt) | `simple_diarizer`를 포함한 서버용 패키지 목록 |
| [`requirements.txt`](./requirements.txt) | 기존 패키지 목록·pyannote 실험 의존성 |
| [`install.sh`](./install.sh) | 당시 Conda·패키지 설치 명령 |
| [`test.ipynb`](./test.ipynb) | 영상 음성 다운로드 실험 |

## References

- [팀 소개와 역할](https://github.com/MettingApp/.github/blob/main/profile/README.md)
- [Mote 발표자료](https://github.com/user-attachments/files/19148912/Mote.pdf): 서비스 구성·모델 파일 용량 비교
- [simple_diarizer](https://github.com/cvqluu/simple_diarizer): x-vector·spectral clustering 기반 화자 분리 라이브러리
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): CTranslate2 기반 Whisper 추론
- [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [GPT-3.5 Turbo 상태](https://developers.openai.com/api/docs/models/gpt-3.5-turbo)
