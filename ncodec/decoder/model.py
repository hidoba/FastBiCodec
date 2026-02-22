import re
import torch
import numpy as np
import onnxruntime as ort
from ncodec.decoder.model_utils import AudioTokenizer

class AudioDecoder:
    def __init__(self, decoder_paths):
        
        sess_options = ort.SessionOptions()
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0})
        ]
        self.processor_detokenizer = ort.InferenceSession(f"{decoder_paths}/processer.onnx", sess_options, providers=providers)
        self.audio_detokenizer = AudioTokenizer(f'{decoder_paths}/detokenizer.safetensors')
        self.output_gain = 1.0

    @torch.inference_mode()
    def detokenize(self, context_tokens, speech_tokens):
        """helper function to detokenize"""
        
        speech_tokens = (
            torch.tensor([int(token) for token in re.findall(r"speech_token_(\d+)", speech_tokens)])
            .long()
            .unsqueeze(0)
        ).numpy()
        context_tokens = (
            torch.tensor([int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)])
            .long()
            .unsqueeze(0).unsqueeze(0)
        ).numpy().astype(np.int32)
        
        x = self.processor_detokenizer.run(["preprocessed_output"], {"context_tokens": context_tokens, "speech_tokens": speech_tokens})
        x = torch.from_numpy(x[0]).to("cuda:0")
        lowres_wav = self.audio_detokenizer.decode(x).squeeze(0)

        if self.output_gain != 1.0:
            lowres_wav = torch.clamp(lowres_wav * self.output_gain, -1.0, 1.0)

        return lowres_wav.cpu()
