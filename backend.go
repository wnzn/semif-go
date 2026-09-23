package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type backend struct {
	base      string
	key       string
	client    *http.Client
	initial   int
	maximum   int
	maxPrompt int
}

type serverProps struct {
	MediaMarker string          `json:"media_marker"`
	Modalities  map[string]bool `json:"modalities"`
}

var errPromptTooLong = errors.New("prompt exceeds token limit")

func newBackend(base, key string, initial, maximum, maxPrompt int) (*backend, error) {
	u, err := url.Parse(base)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("llama-url must be an HTTP(S) base URL")
	}
	return &backend{base: strings.TrimRight(base, "/"), key: key, client: &http.Client{Timeout: 5 * time.Minute}, initial: initial, maximum: maximum, maxPrompt: maxPrompt}, nil
}

func (b *backend) post(ctx context.Context, route string, input, output any) error {
	data, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.base+route, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if b.key != "" {
		req.Header.Set("Authorization", "Bearer "+b.key)
	}
	res, err := b.client.Do(req)
	if err != nil {
		return fmt.Errorf("llama-server %s: %w", route, err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(res.Body, 1024))
		return fmt.Errorf("llama-server %s returned %d: %s", route, res.StatusCode, strings.TrimSpace(string(message)))
	}
	if err := json.NewDecoder(io.LimitReader(res.Body, 128<<20)).Decode(output); err != nil {
		return fmt.Errorf("llama-server %s response: %w", route, err)
	}
	return nil
}

func (b *backend) props(ctx context.Context) (serverProps, error) {
	var props serverProps
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.base+"/props", nil)
	if err != nil {
		return props, err
	}
	if b.key != "" {
		req.Header.Set("Authorization", "Bearer "+b.key)
	}
	res, err := b.client.Do(req)
	if err != nil {
		return props, fmt.Errorf("llama-server /props: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return props, fmt.Errorf("llama-server /props returned %d", res.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(res.Body, 2<<20)).Decode(&props); err != nil {
		return props, fmt.Errorf("llama-server /props response: %w", err)
	}
	if props.MediaMarker == "" {
		return props, errors.New("llama-server did not provide a media_marker in /props")
	}
	return props, nil
}

func (b *backend) render(ctx context.Context, user string) (string, error) {
	input := map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": directSystem},
			{"role": "user", "content": user},
		},
		"add_generation_prompt": true,
		"chat_template_kwargs":  map[string]bool{"enable_thinking": false},
	}
	var result struct {
		Prompt string `json:"prompt"`
	}
	if err := b.post(ctx, "/apply-template", input, &result); err != nil {
		return "", err
	}
	if result.Prompt == "" {
		return "", errors.New("llama-server returned an empty chat template prompt")
	}
	return result.Prompt, nil
}

func (b *backend) tokenize(ctx context.Context, text string) ([]int, error) {
	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := b.post(ctx, "/tokenize", map[string]any{"content": text, "add_special": false, "parse_special": true}, &result); err != nil {
		return nil, err
	}
	return result.Tokens, nil
}

func (b *backend) encode(ctx context.Context, prompt string, count int) ([]int, []int, error) {
	ids, err := b.tokenize(ctx, prompt)
	if err != nil {
		return nil, nil, err
	}
	if len(ids) == 0 || len(ids) > b.maxPrompt {
		return nil, nil, fmt.Errorf("%w: prompt has %d tokens; allowed range is 1-%d", errPromptTooLong, len(ids), b.maxPrompt)
	}
	slots := make([]int, count)
	for i := range slots {
		letter := string(rune('A' + i))
		piece, err := b.tokenize(ctx, letter)
		if err != nil {
			return nil, nil, err
		}
		joined, err := b.tokenize(ctx, prompt+letter)
		if err != nil {
			return nil, nil, err
		}
		if len(piece) != 1 || len(joined) != len(ids)+1 || !equalTokens(ids, joined[:len(ids)]) || joined[len(ids)] != piece[0] {
			return nil, nil, fmt.Errorf("answer slot %s is not one token at the prompt boundary", letter)
		}
		for _, previous := range slots[:i] {
			if previous == piece[0] {
				return nil, nil, errors.New("answer slot tokens collide")
			}
		}
		slots[i] = piece[0]
	}
	return ids, slots, nil
}

func equalTokens(a, b []int) bool {
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

type tokenProbability struct {
	ID      int     `json:"id"`
	Logprob float64 `json:"logprob"`
}

func (b *backend) score(ctx context.Context, prompt any, slots []int) ([]float64, int, int, error) {
	outputTokens := 0
	for top := b.initial; ; {
		var result struct {
			Probabilities []struct {
				Top []tokenProbability `json:"top_logprobs"`
			} `json:"completion_probabilities"`
			Truncated       bool `json:"truncated"`
			TokensEvaluated int  `json:"tokens_evaluated"`
		}
		input := map[string]any{
			"prompt": prompt, "n_predict": 1, "n_probs": top,
			"post_sampling_probs": false, "cache_prompt": true,
			"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
			"repeat_penalty": 1.0, "repeat_last_n": 0,
		}
		if err := b.post(ctx, "/completion", input, &result); err != nil {
			return nil, 0, 0, err
		}
		if result.Truncated || len(result.Probabilities) == 0 || len(result.Probabilities[0].Top) == 0 {
			return nil, 0, 0, errors.New("llama-server truncated the prompt or returned no next-token probabilities")
		}
		outputTokens += len(result.Probabilities)
		lookup := make(map[int]float64, len(result.Probabilities[0].Top))
		for _, candidate := range result.Probabilities[0].Top {
			lookup[candidate.ID] = candidate.Logprob
		}
		values := make([]float64, len(slots))
		found := true
		for i, id := range slots {
			value, ok := lookup[id]
			if !ok {
				found = false
				break
			}
			if math.IsNaN(value) || math.IsInf(value, 0) || value <= -1e30 {
				return nil, 0, 0, fmt.Errorf("answer token %d has a zero or invalid next-token probability", id)
			}
			values[i] = value
		}
		if found {
			return softmax(values), result.TokensEvaluated, outputTokens, nil
		}
		if top >= b.maximum {
			return nil, 0, 0, fmt.Errorf("an answer token is outside the top %d next tokens; raise -max-top-probs", top)
		}
		if top > b.maximum/2 {
			top = b.maximum
		} else {
			top *= 2
		}
	}
}

func softmax(values []float64) []float64 {
	maximum := values[0]
	for _, value := range values[1:] {
		if value > maximum {
			maximum = value
		}
	}
	result := make([]float64, len(values))
	sum := 0.0
	for i, value := range values {
		result[i] = math.Exp(value - maximum)
		sum += result[i]
	}
	for i := range result {
		result[i] /= sum
	}
	return result
}
