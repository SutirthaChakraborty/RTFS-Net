import os
import sys
import torch
import importlib
import torchaudio
from sigfig import round
import yaml
from src.utils.parser_utils import parse_args_as_dict
from src.datas.transform import get_preprocessing_pipelines
import numpy as np
import soundfile as sf
import argparse
import librosa
from RTFSNet_file import get_video_crops
import time

from moviepy.editor import VideoFileClip, AudioFileClip

def add_audio_to_video(video_path, audio_path, output_path):
    # Load the video file
    video_clip = VideoFileClip(video_path)

    # Load the audio file
    audio_clip = AudioFileClip(audio_path)

    # Set the audio of the video clip as the audio file
    video_clip_with_audio = video_clip.set_audio(audio_clip)

    # Write the result to a file
    video_clip_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class TestOneVideo():
    def __init__(self, conf):
        super(TestOneVideo, self).__init__()
        self.conf = conf
        self.conf["videonet"] = conf.get("videonet", {})
        self.conf["videonet"]["model_name"] = conf["videonet"].get("model_name", None)
        self.exp_dir = os.path.abspath(os.path.join("../experiments/audio-visual", conf["log"]["exp_name"]))

        sys.path.append(os.path.dirname(self.exp_dir))
        models_module = importlib.import_module(os.path.basename(self.exp_dir) + ".models")
        videomodels = importlib.import_module(os.path.basename(self.exp_dir) + ".models.videomodels")
        AVNet = getattr(models_module, "AVNet")

        model_path = os.path.join(self.exp_dir, "best_model.pth")
        self.audiomodel = AVNet.from_pretrain(model_path, **self.conf["audionet"])
        self.videomodel = None
        if self.conf["videonet"]["model_name"]:
            self.videomodel = videomodels.get(self.conf["videonet"]["model_name"])(**self.conf["videonet"], print_macs=False)

    def test(self):
        with torch.no_grad():
            for idx in range(1, 2):
                file_name='inference/myfolder/sample.mp4'
                mix, fs = librosa.load(file_name,sr=16000)
                # Resample if the current sample rate is not 16,000 Hz
                
                if fs != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                    mix = resampler(torch.from_numpy(mix)).numpy()
                
                mixture=torch.from_numpy(mix)
                m_std = mixture.std(-1, keepdim=True)
                EPS = 1e-8
                mixture =mixture.squeeze(0)
                
                print("Extracting video Features...")
                target_mouths=torch.from_numpy(get_preprocessing_pipelines()["val"](get_video_crops(file_name))).unsqueeze(0)
                
                print("Get Video Embeddings...")
                start_time = time.time()
                mouth_emb = self.videomodel(target_mouths.float().unsqueeze(0)) if self.videomodel is not None else None
                end_time = time.time() # Current time after execution
                elapsed_time = end_time - start_time
                print(f"Time taken for Video embeddings: {elapsed_time} seconds")
                
                start_time = time.time()
                est_sources = self.audiomodel(mixture.unsqueeze(0), mouth_emb)
                end_time = time.time() # Current time after execution
                elapsed_time = end_time - start_time
                print(f"Time taken for Audio Prediction: {elapsed_time} seconds")
                
                
                est_sources_np = est_sources.cpu().squeeze(0)
                torchaudio.save('infer_pred.wav', est_sources_np, 16000)
                mix_np = mixture.cpu().unsqueeze(0)
                torchaudio.save(os.path.join("infer_mix.wav"), mix_np, 16000)

                # Paths to the video and audio files
                audio_file = 'infer_pred.wav'
                output_file = 'prediction.mp4'

                # Add the audio to the video
                add_audio_to_video(file_name, audio_file, output_file)



def main(conf):
    model = TestOneVideo(conf)
    model.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf-dir",
        type=str,
         default="../experiments/audio-visual/RTFS-Net/LRS2/12_layers/conf.yaml",
        help="Full path to save best validation model",
    )

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic["main_args"])

    main(def_conf)
