import os

output_path = '/home/jay_kejriwal/Fisher/Processed/VACC_audio_resampled'
audio_path = '/home/jay_kejriwal/Fisher/Processed/VACC_audio'
all_files = os.listdir(audio_path)
for root, dirs, files in os.walk(audio_path):
    for file in files:
        if file.endswith('.wav'):
            out_name= os.path.basename(file)
            output=os.path.join(output_path, out_name)
            #cmd_str = f"ffmpeg -i {os.path.join(root, file)} -ac 1 -ar 16000 {output}"
            cmd_str = f"sox {os.path.join(root, file)} -r 16000 {output}"
            print(cmd_str)
            os.system(cmd_str)







