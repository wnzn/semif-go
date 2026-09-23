# SemIf System One adapter (Go)

A standalone HTTP service that scores runtime questions against a GGUF model running in a separate, unmodified `llama-server` process. It uses SemIf's direct-options prompt and conditional answer-letter probabilities. The HTTP request and response shapes follow [TypeSafe's System One API](https://docs.typesafe.ai/api); the model and scoring method are local, not Jev.

Requires Go 1.24+ and a recent llama.cpp server with `/apply-template`, `/tokenize`, and pre-sampling `n_probs` on `/completion` ([llama.cpp server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

## Run

Start llama.cpp with a GGUF checkpoint containing a compatible chat template. For SemIf's published text baseline, use its Qwen3.5-4B GGUF/checkpoint pairing. For media input, start llama-server with a multimodal model and its projector when required by that model. Check `GET /props` for `modalities.vision`, `.audio`, and `.video` before sending media. See the [llama.cpp multimodal guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

```sh
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 1 \
  --n-gpu-layers 0 --threads 8
```

In this directory, start the Go service:

```sh
go run . -listen 127.0.0.1:8090 -llama-url http://127.0.0.1:8080 \
  -model semif-local -max-prompt-tokens 4096
```

`LLAMA_API_KEY` supplies a bearer token for llama-server if it requires one. Set `SEMIF_API_KEY` to require a bearer token from clients of the Go service. The adapter listens on loopback by default.

```sh
curl -sS http://127.0.0.1:8090/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "The customer cannot log in after resetting a password.",
    "questions": {
      "queue": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "access": "Account access support.",
          "billing": "Billing support."
        }
      },
      "urgent": {"type": "noul", "instructions": "Is this urgent?"},
      "severity": {
        "type": "score",
        "instructions": "How serious is this issue?",
        "criteria": ["Minor", "Moderate", "Critical"]
      }
    }
  }'
```

The response contains `model: "semif-local"`, an `answers` object keyed by question ID, and `usage` with input/output token counts. `jev-latest` is accepted as an **input alias** for clients that default to it; this service never returns a Jev model name. Choice returns `choice`, `probabilities` and `confidence`; Noul returns `noul`; Score returns `score`, `legend`, `probabilities` and `confidence`. Invalid requests return 422; failures to obtain a complete score from llama-server return 502.

## Multimodal state

With a suitable model, send an explicit `type: "multimodal"` object as `state`. Text and media are kept in the order given. The API accepts `image_url` with an inline image data URL, and `input_audio` and `input_video` with raw base64 in `data` or a data URL in `url`:

```json
{
  "model": "semif-local",
  "state": {
    "type": "multimodal",
    "content": [
      {"type": "text", "text": "Which product appears in this clip?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_PNG>"}},
      {"type": "input_audio", "input_audio": {"data": "<BASE64_WAV>"}},
      {"type": "input_video", "input_video": {"url": "data:video/mp4;base64,<BASE64_MP4>"}}
    ]
  },
  "questions": {
    "product": {
      "type": "choice",
      "instructions": "Which product is shown?",
      "criteria": {"phone": "A phone", "laptop": "A laptop"}
    }
  }
}
```

For example, `base64 -w0 picture.png` yields the image payload on GNU/Linux. This adapter takes inline bytes only; it does not fetch remote URLs or read server-side files. Requests are capped at 64 MiB, with at most 8 media parts, 32 MiB per part and 48 MiB decoded media total. PDF is not a native llama.cpp media input: extract text or render pages as images first. The actual formats depend on the model and llama-server build; llama.cpp documents images, audio and video input through its [server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Video decoding also requires `ffmpeg` and `ffprobe` on the llama-server host's `PATH`, or a configured `--video-ffmpeg-dir`. Check with `command -v ffmpeg && command -v ffprobe` before trying `input_video`. A model reporting `modalities.video: true` in `/props` does not establish that those executables are installed. They were not on the PATH of the machine used to develop this adapter; image and audio input can still work without them when the model supports those modalities.

The adapter reads llama-server's current media marker and capabilities from `GET /props`, inserts one marker per media part in the question prompt, then sends the rendered prompt and ordered base64 bytes to `/completion` as `prompt_string` plus `multimodal_data`. llama.cpp handles media decoding and inference. Actual multimodal input token counts come from llama-server's `tokens_evaluated` response. Increase `-max-prompt-tokens` for media-heavy inputs; the server context window must also fit them. For text-only state, the original token-ID path is unchanged.

## Scoring details

The prompt uses SemIf's [system instruction and evidence/criterion/options payload](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/src/semif_phase1/core.py), formatted through the GGUF's chat template with thinking disabled. The service tokenizes the rendered prompt and verifies that every `A`-`P` answer slot is exactly one token at the answer boundary, as in [SemIf's direct scorer](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/src/semif_phase1/direct.py).

For each question, the service requests one completion token plus **pre-sampling** next-token log-probabilities. It increases `n_probs` until all option tokens appear, then applies softmax over only those option scores. If the requested top-N ceiling is reached without all options, it returns an error rather than silently assigning zero probability. Configure `-initial-top-probs` and `-max-top-probs` to trade response size against retries (defaults: 256 and 262144). Llama-server's `cache_prompt` may reuse shared state; it is not SemIf's explicit whole-sequence state-restore mode.

Choice supports 2-16 options because SemIf uses the single-token letters `A`-`P`; Score supports 2-10 levels. `confidence` is **this adapter's normalized-entropy statistic** (`1 - H(p)/log(n)`), not TypeSafe's unpublished confidence function or calibrated decision confidence. The option probabilities are conditional on the declared options and are uncalibrated. Score is the probability-weighted average of zero-based level indices. `usage.output_tokens` counts the probe completions needed to retrieve logits.

GGUF chat templates and tokenization can differ from the pinned Transformers reference. Check against reference prompts before relying on matching SemIf/Torch prompt hashes or score parity. This service uses only the GGUF/server tokenizer and does not claim to reproduce Jev's model or training.

Multimodal prompts use an evidence string with media markers, so they are **not** identical to SemIf's text-only prompt. Image/audio/video scoring requires a supporting GGUF and llama-server build; the test suite checks the HTTP integration with a mock server, not live multimodal model quality. [Gemma 4 12B](https://ai.google.dev/gemma/docs/core/model_card_4) is one example of a model with text, image, audio and video-as-frames input, but the adapter does not require Gemma.

This project adapts the direct-options prompt and scoring approach of [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), which is MIT-licensed. It is independent of SemIf, TypeSafe, and Jev.

```sh
go test ./...
go vet ./...
```
