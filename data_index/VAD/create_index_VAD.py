import os
import pandas as pd
import textgrids
import numpy as np
import json


def index(vad_folder_path):

    VAD_path = os.path.join(vad_folder_path, 'data/VAD/Audio')

    #Check Paths
    data = []
    for root, _, files in os.walk(VAD_path):
        for file in files:
            # Exclude hidden files
            if file.startswith('.'):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, vad_folder_path)
            directory = os.path.dirname(rel_path)
            
            if directory:
                dir_parts = directory.split(os.sep)
                if len(dir_parts) > 1:
                    directory, subdirectory = dir_parts[-2], dir_parts[-1]
                else:
                    directory, subdirectory = dir_parts[-2], ""
            else:
                directory, subdirectory = "", ""
            
            # Swap directory and subdirectory if subdirectory is TIMIT or PTDB-TUG
            if subdirectory in ['TMIT', 'PTDB-TUG']:
                directory, subdirectory = subdirectory, directory
            
            #speakers
            if directory in ['PTDB-TUG']:
                speaker = file.split('_')[1]
            elif directory in ['Noizeus']:
                speaker = file.split('_')[0]
            else:
                speaker = file #No info on speaker - It would need to have access to the entire TIMIT corpus to get the speaker info - Asked the access through my university

            data.append({
                'file_path': full_path,
                'corpus': directory,
                'meta': subdirectory,
                'filename': file,
                'speaker': speaker
            })

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(vad_folder_path, 'data_index/VAD/index_VAD.csv'), index=False)

    labels = {'NOSPEECH': 0, 'SPEECH': 1}

    json.dump(labels, open(os.path.join(vad_folder_path, 'data_index/VAD/labels.json'), 'w'))