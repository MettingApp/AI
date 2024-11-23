import os
from tqdm import tqdm
import soundfile as sf
from openai import OpenAI
from simple_diarizer.diarizer import Diarizer
from faster_whisper import WhisperModel

API_KEY = open('API_KEY.txt', 'r').read()
Nicknames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


def speaker_diarization(diar_model, wav_file, num_speakers, output_dir):
    

    segments = diar_model.diarize(wav_file, num_speakers=num_speakers)

    signal, fs = sf.read(wav_file)
    
    os.makedirs(output_dir, exist_ok=True)

    for seg in segments:

        if seg['end'] - seg['start'] < 1.0:
            continue

        start = seg["start_sample"]
        end = seg["end_sample"]
        speech = signal[start:end]

        filename = f"{int(seg['start'])}_{int(seg['end'])}_speaker_{seg['label']}.wav"
        file_path = os.path.join(output_dir, filename)

        sf.write(file_path, speech, fs)
        # print(f"Speaker {seg['label']} ({seg['start']}s - {seg['end']}s) saved to {file_path}")
        
    return [os.path.join(output_dir, file) for file in sorted(os.listdir(output_directory))]

def stt_transcription(stt_model, wav_file_list):
    
    with open('RESULT.txt', 'wt', encoding='utf-8') as f:
        
        for file in tqdm(wav_file_list):
            
            nick = Nicknames[int(file.split('_')[-1][:-4])]
            
            segments, info = stt_model.transcribe(file, language='ko')
            
            for segment in segments:
                m, s = divmod(segment.start, 60)
                time_str = f"{int(m):02d}:{int(s):02d}" if m > 0 else f"{int(s):02d}"
                f.write(f"[{time_str}] {nick}: {segment.text}\n")
                # print(f"{nick}: {segment.text}")
        
        f.close()

        
def llm_summarization(client, text_file):
    
    chat_completion = client.chat.completions.create(
        messages=[
            { 'role': "system", 'content': "You are a assistant to clarify the meeting in a meeting script." },
            {
                "role": "user",
                "content": 
                    f"""회의 대본 : {open(text_file, 'r').read()}
                    회의 대본은 회의 음성을 텍스트로 변환한 거야. 다음 지침을 따라줘. 
                    1. 한국어로 요약해줘.         
                    2. 회의 대본에서 각 주제별로 요약하고 결론을 자세히 알려줘. 화자는 언급하지 않아도 돼.
                    3. 회의 대본에서 더 조사하려는 내용이 있었으면 해당 내용의 검색 키워드를 알려줘.
                    4. 회의 내용 중 다음 회의 일정을 언급했다면 미팅 예정일과 시간, 장소를 알려줘.
                    5. 아래 출력 형식으로 Markdown code를 작성해줘.
                    6. 출력 형식 : '
                    (제목2)topic summary: 
                        1. 주제 1 
                            * summry : 회의 내용 요약
                            * result : 회의 결론
                            * search : (이텔릭체)추가로 조사할 내용에 대한 검색 문장
                        2. 주제 2 
                            * summary : 회의 내용 요약
                            * result : 회의 결론
                            * search : (이텔릭체)추가로 조사할 내용에 대한 검색 문장
                    (인용)next meeting: (날짜(월/일 or 이번주 금요일), 시간(오전/오후 n시 or 미정), 장소(없을 경우 미정) or 없음
                    '
                    """
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0.2,
    )
    
    return chat_completion.choices[0].message.content

        
def main(wav_file, num_speakers, organization_field):
    diar_model = Diarizer(
                  embed_model='xvec', # 'xvec' and 'ecapa' supported
                  cluster_method='sc' # 'ahc' and 'sc' supported
               )
    stt_model = WhisperModel("imTak/faster-whisper_Korean_L3turbo", device="cpu")
    client = OpenAI(api_key=API_KEY)
    
    # speaker diarization
    wav_file_list = speaker_diarization(diar_model, wav_file, num_speakers, "speaker_diarization")
    
    # stt transcription
    stt_transcription(stt_model, wav_file_list)
    
    # llm summarization
    md_code = llm_summarization(client, "RESULT.txt")
    
    print(md_code)
        
if __name__ == '__main__':
    main('example/XtVE-9ywfDc.wav', 5, 'Culture')
