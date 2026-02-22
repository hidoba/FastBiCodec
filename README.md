# FastBicodec (hidoba fork)

Optimized version of Bicodec. Forked from [ysharma3501/FastBiCodec](https://github.com/ysharma3501/FastBiCodec).

## Changes in this fork

- **Removed FastAudioSR upsampler dependency** — the neural upsampler (16kHz → 48kHz) applied adaptive normalization that was inconsistent across streaming chunks, making crossfade blending difficult.
- **Added configurable fixed output gain** — `AudioDecoder.output_gain` (default 1.0) applies a simple multiplicative gain with clamping to [-1, 1]. Ensures consistent volume across all decoded chunks.
- **Output is native 16kHz** — resampling to the target sample rate is handled externally (e.g. via `torchaudio.transforms.Resample`).
