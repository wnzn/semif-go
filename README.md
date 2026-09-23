# semif-go

**Turn a compatible local LLM into a Jev-like decision API.** Bring your own GGUF model and run it with [llama.cpp](https://github.com/ggml-org/llama.cpp); this lightweight Go service accepts a state and runtime-defined questions, then returns typed Choice, Noul (yes/no), and Score answers at `POST /v1/systemone`. No answer sentence or JSON generation is needed: it scores the model's next-token probabilities over the available options.

With a multimodal model, the same API can evaluate **text, images, audio, and video** in one state. The Go service forwards media to llama.cpp for decoding and inference. Use a text-only model for text decisions or a vision/audio/video-capable model for the corresponding media; the service is not tied to Gemma, Qwen, or any one checkpoint.

This is an **independent, open-source adapter**, inspired by [Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and built on the direct-option scoring method of [SemIf (formerly OpenJev)](https://github.com/TheoLeeCJ/SemIf-OpenJev). It is not Jev, does not use Jev's model or training, and is not affiliated with TypeSafe or SemIf.

## How it works

```text
Your app  -- state + typed questions -->  semif-go :8090
                                         | render prompt, verify option tokens
                                         v
                                    llama-server :8080  -->  GGUF model
                                         | next-token probabilities
                                         v
Your app  <-- Choice / Noul / Score --  semif-go
```

Each option is assigned a single uppercase letter (`A` through `P`). The adapter verifies those token IDs at the prompt boundary, requests pre-sampling next-token log-probabilities from llama.cpp, and normalizes *only the listed option scores*. Media stays with llama.cpp; Go transports inline bytes and builds the question prompt. The model handles interpretation, so decision quality depends on the model and task.

## Quick start

You need **Go 1.24+**, a GGUF model, and a recent `llama-server` with `/apply-template`, `/tokenize`, `/completion` (`n_probs`), and `/props` for multimodal requests ([server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Start your existing server or launch one like this:

```sh
llama-server \
  --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 1
```

For a model that needs a multimodal projector, add `--mmproj /path/to/mmproj.gguf`. Check the loaded model's capabilities with `curl -sS http://127.0.0.1:8080/props`; the `modalities` object reports `vision`, `audio`, and `video`. See the [llama.cpp multimodal guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

In this repository, run the adapter on a different port:

```sh
go run . -listen 127.0.0.1:8090 \
  -llama-url http://127.0.0.1:8080 \
  -model my-local-model -max-prompt-tokens 8000
```

`-model` is the label returned in responses, **not** a command to load or change the model in llama.cpp. Set `LLAMA_API_KEY` if your llama-server requires a bearer token; set `SEMIF_API_KEY` to require one from adapter clients. The adapter listens on loopback by default.

### Ask several questions at once

```sh
curl -sS http://127.0.0.1:8090/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-local-model",
    "state": "I was charged twice for my subscription. Please refund the duplicate payment.",
    "questions": {
      "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "access": "Account login and recovery",
          "billing": "Invoices, payments, and refunds",
          "technical": "Bugs and outages",
          "shipping": "Delivery and tracking"
        }
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?"
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["No urgency", "Time-sensitive", "Immediate action required"]
      }
    }
  }'
```

The response has `model`, question-keyed `answers`, and token `usage`. Choice includes a selected option, all option probabilities, and confidence; Noul is the probability of yes; Score includes a probability-weighted zero-based level, a legend, level probabilities, and confidence. The request also accepts `"jev-latest"` as an **input alias** for clients that default to it, but the response reports your configured *local* model label. See the [System One HTTP reference](https://docs.typesafe.ai/api) for the interface pattern; this service is not a behavioral clone of Jev.

## Multimodal state

Use an explicit `type: "multimodal"` state with ordered content parts. You can interleave text with images, audio, and video, subject to the model's reported capabilities:

```json
{
  "model": "my-local-model",
  "state": {
    "type": "multimodal",
    "content": [
      {"type": "text", "text": "What product is shown or mentioned?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_PNG>"}},
      {"type": "input_audio", "input_audio": {"data": "<BASE64_WAV>"}},
      {"type": "input_video", "input_video": {"url": "data:video/mp4;base64,<BASE64_MP4>"}}
    ]
  },
  "questions": {
    "product": {
      "type": "choice",
      "instructions": "Which product appears in the evidence?",
      "criteria": {"phone": "A phone", "laptop": "A laptop", "neither": "Neither"}
    }
  }
}
```

You can send just one media part; the example shows all three kinds together. Image parts use inline `data:image/...;base64,...` URLs. Audio and video parts accept raw base64 in `data` or a matching `data:audio/...;base64,...` / `data:video/...;base64,...` URL in `url`. On GNU/Linux, `base64 -w0 picture.png` creates an image payload. Remote URLs and server-side file paths are **not** fetched by this adapter. Existing string/object/array states remain text-only unless explicitly marked `type: "multimodal"`.

The adapter obtains llama-server's media marker and capabilities from `GET /props`, places markers in the rendered prompt, then submits `prompt_string` and ordered `multimodal_data` to `/completion`. llama.cpp decodes the media and runs the model. Multimodal token usage comes from llama-server's `tokens_evaluated`. Increase `-max-prompt-tokens` and the server's `--ctx-size` as needed for media-heavy prompts.

**Limits and formats:** at most 8 media parts, 32 MiB decoded per part, 48 MiB decoded in total, and a 64 MiB HTTP request body. Supported image/audio/video formats depend on your llama.cpp build and the loaded model. Video decoding additionally needs `ffmpeg` and `ffprobe` on the llama-server host's `PATH`, or `--video-ffmpeg-dir`; `modalities.video: true` alone does not confirm those tools are installed. **PDFs are not a native input here:** extract their text or render pages to images before sending them. [llama.cpp media formats](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

## Compatibility and score interpretation

- The local GGUF model must provide a usable chat template and distinct single-token answer letters at the prompt boundary. Unsupported tokenizers fail rather than return invented option scores.
- Choice supports **2–16 options** (`A`–`P`); Score supports **2–10 levels**. Several questions may share a state, but they are scored separately; llama-server may reuse a prompt prefix through `cache_prompt`.
- If an answer token is not in llama-server's top `n_probs` results, the adapter retries with a larger top-N. Configure `-initial-top-probs` (default 256) and `-max-top-probs` (default 262144). It returns an error if the full option set still cannot be scored.
- Probabilities are **conditional on the listed options and uncalibrated**. Choice/Score `confidence` is this adapter's normalized-entropy statistic, not TypeSafe's confidence calculation. Validate decisions on your own workload before using confidence thresholds.
- Text prompts follow [SemIf's direct-options system instruction and payload](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/src/semif_phase1/core.py); multimodal evidence is rendered with media markers and has a different prompt. GGUF chat templates/tokenizers can differ from the pinned Transformers reference, so identical SemIf/Torch scores are not guaranteed.

Malformed requests return HTTP 422; upstream failures or missing option probabilities return HTTP 502. One completion-token probe is requested per scoring attempt to obtain the logits; the model's generated text is not used as the answer.

## Credits and research foundations

This project is possible because of earlier work by:

- **[TheoLeeCJ and the SemIf contributors](https://github.com/TheoLeeCJ/SemIf-OpenJev)** — SemIf (formerly OpenJev) demonstrated direct option-logit readout from open models and provided the [prompt and llama.cpp scoring method](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/src/semif_phase1/llamacpp_backend.py) adapted here. SemIf is MIT-licensed; its notice is retained in this repository's [LICENSE](LICENSE).
- **[Diogo Almeida and the TypeSafe AI team](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** — Jev and the System One concept of runtime-defined, typed probabilistic decisions inspired the API shape. This project does **not** implement their proprietary model, training, calibration, or serving architecture.
- **[Georgi Gerganov and the llama.cpp contributors](https://github.com/ggml-org/llama.cpp)** — local GGUF inference, the server's next-token probabilities, and multimodal processing that the Go adapter uses.
- **Open-model researchers and creators**, including the [Qwen team](https://huggingface.co/Qwen/Qwen3.5-4B) behind SemIf's original baseline and [Google DeepMind's Gemma team](https://ai.google.dev/gemma/docs/core/model_card_4) behind an example multimodal model. Bring your own compatible model; no model weights are included here.

SemIf, Jev, TypeSafe, llama.cpp, Qwen, and Gemma belong to their respective projects and owners. `semif-go` is independent of and not endorsed by any of them. Project code is released under the [MIT License](LICENSE); your chosen model's license still applies.

## Verify

```sh
go test ./...
go vet ./...
```

Tests exercise the adapter with a mock llama-server. Text and image scoring have also been exercised against a local Gemma 4 12B server; results and speed depend on the model and hardware. Audio/video decoding has not been live-verified by this repository.
