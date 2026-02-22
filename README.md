# FastBicodec (hidoba fork)

Optimized version of Bicodec. Forked from [ysharma3501/FastBiCodec](https://github.com/ysharma3501/FastBiCodec).

## Changes in this fork

- **Removed FastAudioSR upsampler dependency** — the neural upsampler (16kHz → 48kHz) applied adaptive normalization that was inconsistent across streaming chunks, making crossfade blending difficult.
- **Added configurable fixed output gain** — `AudioDecoder.output_gain` (default 1.0) applies a simple multiplicative gain with clamping to [-1, 1]. Ensures consistent volume across all decoded chunks.
- **Output is native 16kHz** — resampling to the target sample rate is handled externally (e.g. via `torchaudio.transforms.Resample`).
- **Optional local model path** — `TTSCodec(model_path="/path/to/MiraTTS")` skips `snapshot_download` and uses a local directory. Falls back to downloading from HuggingFace if the path is not a valid directory or not provided.
- **Optional encoder loading** — `TTSCodec(load_encoder=False)` skips loading the `AudioEncoder` (wav2vec2 + ONNX quantizers, ~1.3GB download). Use this when you only need decoding and voice embeddings are pre-extracted.
