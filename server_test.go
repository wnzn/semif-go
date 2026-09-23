package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func mockLlama(t *testing.T, complete func(http.ResponseWriter, *http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/props":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"media_marker": "<__media_test__>",
				"modalities":   map[string]bool{"vision": true, "audio": true, "video": true},
			})
		case "/apply-template":
			var req struct {
				Messages []struct {
					Content string `json:"content"`
				} `json:"messages"`
				Kwargs map[string]bool `json:"chat_template_kwargs"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) != 2 || req.Kwargs["enable_thinking"] {
				t.Errorf("invalid template request: %+v, %v", req, err)
			}
			_, _ = fmt.Fprintf(w, `{"prompt":%q}`, "<assistant>"+req.Messages[1].Content+"<answer>")
		case "/tokenize":
			var req struct {
				Content    string `json:"content"`
				AddSpecial bool   `json:"add_special"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.AddSpecial {
				t.Errorf("invalid tokenization request: %+v, %v", req, err)
			}
			ids := make([]int, 0, len(req.Content))
			for _, ch := range req.Content {
				ids = append(ids, int(ch))
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"tokens": ids})
		case "/completion":
			complete(w, r)
		default:
			t.Errorf("unexpected upstream path %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
}

func makeService(t *testing.T, upstream string, initial, maximum int) *service {
	t.Helper()
	b, err := newBackend(upstream, "", initial, maximum, 4096)
	if err != nil {
		t.Fatal(err)
	}
	return &service{backend: b, model: "semif-local"}
}

func TestMixedSystemOneRequest(t *testing.T) {
	var mu sync.Mutex
	var prompts []string
	upstream := mockLlama(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt      []int `json:"prompt"`
			NPredict    int   `json:"n_predict"`
			NProbs      int   `json:"n_probs"`
			PostSample  bool  `json:"post_sampling_probs"`
			CachePrompt bool  `json:"cache_prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.NPredict != 1 || req.PostSample || !req.CachePrompt {
			t.Errorf("bad completion request: %+v, %v", req, err)
		}
		prompt := stringify(req.Prompt)
		mu.Lock()
		prompts = append(prompts, prompt)
		mu.Unlock()
		if !strings.Contains(prompt, `"evidence": {"message": "urgent"}`) {
			t.Errorf("lost structured state: %s", prompt)
		}
		options := []map[string]any{{"id": int('A'), "logprob": -0.1}, {"id": int('B'), "logprob": -2.1}, {"id": int('C'), "logprob": -3.1}}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"completion_probabilities": []any{map[string]any{"top_logprobs": options}},
			"tokens":                   []int{int('A')}, "truncated": false,
		})
	})
	defer upstream.Close()
	adapter := makeService(t, upstream.URL, 4, 8)
	request := `{"model":"jev-latest","state":{"message":"urgent"},"questions":{` +
		`"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","technical":"Outages"}},` +
		`"urgent":{"type":"noul","instructions":"Is this urgent?"},` +
		`"severity":{"type":"score","instructions":"How severe?","criteria":["low","medium","high"]}}}`
	res := httptest.NewRecorder()
	adapter.evaluate(res, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(request)))
	if res.Code != http.StatusOK {
		t.Fatalf("status %d: %s", res.Code, res.Body.String())
	}
	var reply struct {
		Model   string         `json:"model"`
		Usage   map[string]int `json:"usage"`
		Answers map[string]struct {
			Type          string             `json:"type"`
			Choice        string             `json:"choice"`
			Noul          float64            `json:"noul"`
			Score         float64            `json:"score"`
			Legend        map[string]string  `json:"legend"`
			Probabilities map[string]float64 `json:"probabilities"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &reply); err != nil {
		t.Fatal(err)
	}
	if reply.Model != "semif-local" || reply.Usage["input_tokens"] < 10 || reply.Usage["output_tokens"] != 3 {
		t.Fatalf("bad model/usage: %+v", reply)
	}
	if reply.Answers["team"].Choice != "billing" || !(reply.Answers["team"].Probabilities["billing"] > reply.Answers["team"].Probabilities["technical"]) {
		t.Fatalf("bad choice: %+v", reply.Answers["team"])
	}
	if math.Abs(reply.Answers["urgent"].Noul-0.880797) > 0.00001 {
		t.Fatalf("bad noul: %+v", reply.Answers["urgent"])
	}
	if reply.Answers["severity"].Legend["2"] != "high" || !(reply.Answers["severity"].Score < 0.3) {
		t.Fatalf("bad score: %+v", reply.Answers["severity"])
	}
	mu.Lock()
	defer mu.Unlock()
	if len(prompts) != 3 || !strings.Contains(prompts[1], `"options": [{"letter": "A", "description": "Payments"}, {"letter": "B", "description": "Outages"}]`) {
		t.Fatalf("wrong ordered options: %+v", prompts)
	}
}

func stringify(ids []int) string {
	runes := make([]rune, len(ids))
	for i, id := range ids {
		runes[i] = rune(id)
	}
	return string(runes)
}

func TestRetriesMissingAnswerTokenAndRejectsIncompleteScores(t *testing.T) {
	var requested []int
	upstream := mockLlama(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			NProbs int `json:"n_probs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		requested = append(requested, req.NProbs)
		candidates := []map[string]any{{"id": int('A'), "logprob": -1.0}, {"id": 999, "logprob": -1.5}}
		if req.NProbs >= 4 {
			candidates = append(candidates, map[string]any{"id": int('B'), "logprob": -2.0})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"completion_probabilities": []any{map[string]any{"top_logprobs": candidates}}, "tokens": []int{65},
		})
	})
	defer upstream.Close()
	request := `{"model":"jev-latest","state":"text","questions":{"q":{"type":"choice","instructions":"Choose","criteria":{"a":"A","b":"B"}}}}`
	for _, maximum := range []int{4, 2} {
		requested = nil
		adapter := makeService(t, upstream.URL, 2, maximum)
		res := httptest.NewRecorder()
		adapter.evaluate(res, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(request)))
		if maximum == 4 && (res.Code != http.StatusOK || len(requested) != 2 || requested[1] != 4) {
			t.Fatalf("retry failed: %d %v %s", res.Code, requested, res.Body.String())
		}
		if maximum == 2 && (res.Code != http.StatusBadGateway || !strings.Contains(res.Body.String(), "outside the top 2")) {
			t.Fatalf("should reject incomplete probabilities: %d %s", res.Code, res.Body.String())
		}
	}
}

func TestInputValidation(t *testing.T) {
	adapter := &service{model: "semif-local"}
	for _, test := range []struct {
		name string
		body string
	}{
		{"empty state", `{"model":"jev-latest","state":"","questions":{"x":{"type":"noul","instructions":"hi"}}}`},
		{"unsupported model", `{"model":"other","state":"text","questions":{"x":{"type":"noul","instructions":"hi"}}}`},
		{"too many levels", `{"model":"jev-latest","state":"text","questions":{"x":{"type":"score","instructions":"hi","criteria":[0,1,2,3,4,5,6,7,8,9,10]}}}`},
		{"duplicate choice", `{"model":"jev-latest","state":"text","questions":{"x":{"type":"choice","instructions":"hi","criteria":{"a":"x","a":"y"}}}}`},
		{"extra JSON", `{"model":"jev-latest","state":"text","questions":{}} {}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			res := httptest.NewRecorder()
			adapter.evaluate(res, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(test.body)))
			if res.Code != http.StatusUnprocessableEntity {
				t.Fatalf("status %d: %s", res.Code, res.Body.String())
			}
		})
	}
}

func TestPromptMatchesSemIfDirectMessages(t *testing.T) {
	state := json.RawMessage(`{"text": "hello <世界>", "count": 2}`)
	q := json.RawMessage(`{"type":"choice","instructions":"Who?","criteria":{"first":"A, B","second":"C: D"}}`)
	got, err := prepare(state, q)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"evidence": {"text": "hello <世界>", "count": 2}, "criterion": "Who?", "options": [{"letter": "A", "description": "A, B"}, {"letter": "B", "description": "C: D"}]}`
	if got.Prompt != want {
		t.Fatalf("prompt mismatch:\ngot:  %s\nwant: %s", got.Prompt, want)
	}
}

func TestPromptLimitReturnsValidationError(t *testing.T) {
	upstream := mockLlama(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("completion must not run for an overlong prompt")
	})
	defer upstream.Close()
	adapter := makeService(t, upstream.URL, 4, 8)
	adapter.backend.maxPrompt = 5
	request := `{"model":"jev-latest","state":"text","questions":{"x":{"type":"noul","instructions":"Is this true?"}}}`
	res := httptest.NewRecorder()
	adapter.evaluate(res, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(request)))
	if res.Code != http.StatusUnprocessableEntity || !strings.Contains(res.Body.String(), "prompt exceeds token limit") {
		t.Fatalf("status %d: %s", res.Code, res.Body.String())
	}
}

func TestMultimodalStateForwardsMediaAndScores(t *testing.T) {
	const image = "aW1hZ2U="
	const audio = "YXVkaW8="
	const video = "dmlkZW8="
	var prompts []string
	upstream := mockLlama(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt struct {
				PromptString string   `json:"prompt_string"`
				Media        []string `json:"multimodal_data"`
			} `json:"prompt"`
			NProbs       int  `json:"n_probs"`
			PostSampling bool `json:"post_sampling_probs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			http.Error(w, "invalid completion request", http.StatusBadRequest)
			return
		}
		prompts = append(prompts, req.Prompt.PromptString)
		if req.PostSampling || !strings.Contains(req.Prompt.PromptString, `Image 1: <__media_test__>\nAudio 2: <__media_test__>\nVideo 3: <__media_test__>\n`) || strings.Count(req.Prompt.PromptString, "<__media_test__>") != 3 || !equalStrings(req.Prompt.Media, []string{image, audio, video}) {
			t.Errorf("wrong multimodal prompt: %+v", req)
		}
		if strings.Contains(req.Prompt.PromptString, image) || strings.Contains(req.Prompt.PromptString, audio) {
			t.Errorf("base64 leaked into prompt text: %s", req.Prompt.PromptString)
		}
		candidates := []map[string]any{{"id": int('A'), "logprob": -1.0}, {"id": int('B'), "logprob": -2.0}}
		if req.NProbs == 2 {
			candidates = candidates[:1]
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"completion_probabilities": []any{map[string]any{"top_logprobs": candidates}},
			"tokens_evaluated":         615,
		})
	})
	defer upstream.Close()
	adapter := makeService(t, upstream.URL, 2, 4)
	request := `{"model":"jev-latest","state":{"type":"multimodal","content":[` +
		`{"type":"text","text":"Please classify this"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + image + `"}},` +
		`{"type":"input_audio","input_audio":{"data":"` + audio + `"}},` +
		`{"type":"input_video","input_video":{"url":"data:video/mp4;base64,` + video + `"}}` +
		`]},"questions":{"issue":{"type":"choice","instructions":"What happened?","criteria":{"a":"First","b":"Second"}}}}`
	res := httptest.NewRecorder()
	adapter.evaluate(res, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(request)))
	if res.Code != http.StatusOK {
		t.Fatalf("status %d: %s", res.Code, res.Body.String())
	}
	var result struct {
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
		Answers map[string]struct {
			Choice string `json:"choice"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokens != 615 || result.Answers["issue"].Choice != "a" || len(prompts) != 2 {
		t.Fatalf("unexpected result: %+v, completion calls %d", result, len(prompts))
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestMultimodalValidation(t *testing.T) {
	props := serverProps{MediaMarker: "<__media_test__>", Modalities: map[string]bool{"vision": true, "audio": true, "video": true}}
	for _, test := range []struct{ name, state string }{
		{"remote image", `{"type":"multimodal","content":[{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}`},
		{"bad base64", `{"type":"multimodal","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%%%"}}]}`},
		{"PDF", `{"type":"multimodal","content":[{"type":"application/pdf","data":"YQ=="}]}`},
		{"marker in text", `{"type":"multimodal","content":[{"type":"text","text":"<__media_test__>"},{"type":"input_audio","input_audio":{"data":"YQ=="}}]}`},
		{"text only", `{"type":"multimodal","content":[{"type":"text","text":"hello"}]}`},
		{"empty", `{"type":"multimodal","content":[]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseMediaState(json.RawMessage(test.state), props); err == nil {
				t.Fatalf("expected error for %s", test.state)
			}
		})
	}
	props.Modalities["vision"] = false
	_, err := parseMediaState(json.RawMessage(`{"type":"multimodal","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]}`), props)
	if err == nil || !strings.Contains(err.Error(), "does not support vision") {
		t.Fatalf("missing capability should be rejected: %v", err)
	}
}
