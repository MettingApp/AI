from vctube import VCtube
from datasets import Dataset, Audio, DatasetDict, load_dataset, concatenate_datasets
import pandas as pd
import os

def youtube_to_datasets(playlist_name, playlist_url, lang):
    vc = VCtube(playlist_name, playlist_url, lang)
    vc.download_audio()
    vc.download_captions()
    vc.audio_split()

    df = pd.read_csv(f'datasets/{playlist_name}/text/subtitle.csv')
    ds = Dataset.from_pandas(df).remove_columns(['Unnamed: 0', 'id', 'start', 'duration']).cast_column("name", Audio(sampling_rate=16000))
        
    return ds

lang = 'ko'
# original_dataset = load_dataset('imTak/korean-speeck-Develop', split='train')
original_dataset = None
video_list = ['https://youtu.be/EqoU1PodQQ4?si=bix2WZMdNKP22PQU','https://youtu.be/Tt_tKhhhJqY?si=esuEnXWt1GNB39dv', 'https://youtu.be/M8UPyeF5DfM?si=habdgQYiABYRGuQ3']

# playlist_url =   'https://youtu.be/Tt_tKhhhJqY?si=esuEnXWt1GNB39dv
for video_url in video_list:
    playlist_url = video_url
    playlist_name = playlist_url.split('/')[-1].split('?')[0]
    
    ds = youtube_to_datasets(playlist_name, playlist_url, lang)

    if original_dataset is None:
        original_dataset = ds
        continue
    else:
        updated_dataset = concatenate_datasets([original_dataset, ds])
        original_dataset = updated_dataset


train_test = updated_dataset.train_test_split(test_size=0.2, shuffle=True)

dataset_dict = DatasetDict({'train': train_test['train'], 'test': train_test['test']})

dataset_dict.push_to_hub('korean-speak-Develop')
