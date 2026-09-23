package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
)

const directSystem = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."

type service struct {
	backend *backend
	model   string
	apiKey  string
}

type evaluation struct {
	State     json.RawMessage            `json:"state"`
	Model     string                     `json:"model"`
	Questions map[string]json.RawMessage `json:"questions"`
}

type question struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
}

type option struct {
	ID          string
	Description string
}

type decision struct {
	Kind    string
	Options []option
	Legend  map[string]json.RawMessage
	Prompt  string
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, text string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": text}})
}

func (s *service) evaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	if s.apiKey != "" && r.Header.Get("Authorization") != "Bearer "+s.apiKey {
		writeError(w, http.StatusUnauthorized, "invalid API key")
		return
	}
	var req evaluation
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<20))
	if err := decoder.Decode(&req); err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid JSON request: "+err.Error())
		return
	}
	if decoder.Decode(new(any)) != io.EOF {
		writeError(w, http.StatusUnprocessableEntity, "request must contain one JSON object")
		return
	}
	if err := validateValue(req.State); err != nil {
		writeError(w, http.StatusUnprocessableEntity, "state: "+err.Error())
		return
	}
	if req.Model != "jev-latest" && req.Model != s.model {
		writeError(w, http.StatusUnprocessableEntity, "model must be jev-latest or "+s.model)
		return
	}
	state := req.State
	var media mediaState
	if isMultimodal(state) {
		props, err := s.backend.props(r.Context())
		if err != nil {
			writeError(w, http.StatusBadGateway, err.Error())
			return
		}
		media, err = parseMediaState(state, props)
		if err != nil {
			writeError(w, http.StatusUnprocessableEntity, "state: "+err.Error())
			return
		}
		state = media.Evidence
	}
	if len(req.Questions) == 0 {
		writeError(w, http.StatusUnprocessableEntity, "questions must be a nonempty object")
		return
	}
	keys := make([]string, 0, len(req.Questions))
	prepared := make(map[string]decision, len(req.Questions))
	for name, raw := range req.Questions {
		if strings.TrimSpace(name) == "" {
			writeError(w, http.StatusUnprocessableEntity, "question IDs cannot be empty")
			return
		}
		item, err := prepare(state, raw)
		if err != nil {
			writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("questions[%q]: %s", name, err))
			return
		}
		if len(media.Data) > 0 && strings.Count(item.Prompt, media.Marker) != len(media.Data) {
			writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("questions[%q]: prompt contains unexpected media markers", name))
			return
		}
		prepared[name] = item
		keys = append(keys, name)
	}
	sort.Strings(keys)
	answers := make(map[string]any, len(keys))
	inputTokens, outputTokens := 0, 0
	for _, name := range keys {
		item := prepared[name]
		prompt, err := s.backend.render(r.Context(), item.Prompt)
		if err != nil {
			writeError(w, http.StatusBadGateway, fmt.Sprintf("questions[%q]: %s", name, err))
			return
		}
		if len(media.Data) > 0 && strings.Count(prompt, media.Marker) != len(media.Data) {
			writeError(w, http.StatusBadGateway, fmt.Sprintf("questions[%q]: chat template changed the number of media markers", name))
			return
		}
		ids, slots, err := s.backend.encode(r.Context(), prompt, len(item.Options))
		if err != nil {
			status := http.StatusBadGateway
			if errors.Is(err, errPromptTooLong) {
				status = http.StatusUnprocessableEntity
			}
			writeError(w, status, fmt.Sprintf("questions[%q]: %s", name, err))
			return
		}
		var scoringPrompt any = ids
		if len(media.Data) > 0 {
			scoringPrompt = map[string]any{"prompt_string": prompt, "multimodal_data": media.Data}
		}
		probabilities, counted, output, err := s.backend.score(r.Context(), scoringPrompt, slots)
		if err != nil {
			writeError(w, http.StatusBadGateway, fmt.Sprintf("questions[%q]: %s", name, err))
			return
		}
		if len(media.Data) > 0 && (counted <= 0 || counted > s.backend.maxPrompt) {
			writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("questions[%q]: multimodal prompt has %d tokens; allowed range is 1-%d", name, counted, s.backend.maxPrompt))
			return
		}
		if counted > 0 {
			inputTokens += counted
		} else {
			inputTokens += len(ids)
		}
		outputTokens += output
		answers[name] = item.answer(probabilities)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model": s.model, "answers": answers,
		"usage": map[string]int{"input_tokens": inputTokens, "output_tokens": outputTokens},
	})
}

func validateValue(raw json.RawMessage) error {
	if !json.Valid(raw) {
		return errors.New("must be JSON")
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	switch v := value.(type) {
	case string:
		if strings.TrimSpace(v) != "" {
			return nil
		}
	case []any:
		if len(v) != 0 {
			return nil
		}
	case map[string]any:
		if len(v) != 0 {
			return nil
		}
	}
	return errors.New("must be a nonempty string, object, or array")
}

func describe(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var compact bytes.Buffer
	_ = json.Compact(&compact, raw)
	return compact.String()
}

func prepare(state, raw json.RawMessage) (decision, error) {
	var q question
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '{' || json.Unmarshal(raw, &q) != nil {
		return decision{}, errors.New("question must be an object")
	}
	if err := validateValue(q.Instructions); err != nil {
		return decision{}, fmt.Errorf("instructions: %w", err)
	}
	d := decision{Kind: q.Type}
	switch q.Type {
	case "choice":
		entries, err := orderedCriteria(q.Criteria)
		if err != nil {
			return d, err
		}
		if len(entries) < 2 || len(entries) > 16 {
			return d, errors.New("choice requires 2-16 options (SemIf's single-token letter limit)")
		}
		for _, entry := range entries {
			if strings.TrimSpace(entry.ID) == "" {
				return d, errors.New("choice option IDs must not be empty")
			}
			if string(entry.Value) == "null" {
				d.Options = append(d.Options, option{entry.ID, entry.ID})
				continue
			}
			if err := validateValue(entry.Value); err != nil {
				return d, fmt.Errorf("choice option %q: %w", entry.ID, err)
			}
			d.Options = append(d.Options, option{entry.ID, describe(entry.Value)})
		}
	case "noul":
		d.Options = []option{{"true", "Yes."}, {"false", "No."}}
		if len(q.Criteria) != 0 && string(q.Criteria) != "null" {
			var criteria map[string]json.RawMessage
			if err := json.Unmarshal(q.Criteria, &criteria); err != nil || criteria == nil {
				return d, errors.New("noul criteria must be an object")
			}
			for i, id := range []string{"true", "false"} {
				if rubric, ok := criteria[id]; ok {
					if err := validateValue(rubric); err != nil {
						return d, fmt.Errorf("noul criteria %q: %w", id, err)
					}
					d.Options[i].Description += " " + describe(rubric)
				}
			}
		}
	case "score":
		var levels []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &levels); err != nil || len(levels) < 2 || len(levels) > 10 {
			return d, errors.New("score criteria must be an array of 2-10 levels")
		}
		d.Legend = make(map[string]json.RawMessage, len(levels))
		for i, level := range levels {
			if err := validateValue(level); err != nil {
				return d, fmt.Errorf("score level %d: %w", i, err)
			}
			id := fmt.Sprint(i)
			d.Legend[id] = level
			d.Options = append(d.Options, option{id, describe(level)})
		}
	default:
		return d, errors.New("type must be choice, noul, or score")
	}
	var payload struct {
		Evidence  json.RawMessage `json:"evidence"`
		Criterion string          `json:"criterion"`
		Options   []struct {
			Letter      string `json:"letter"`
			Description string `json:"description"`
		} `json:"options"`
	}
	payload.Evidence = state
	payload.Criterion = describe(q.Instructions)
	for i, opt := range d.Options {
		payload.Options = append(payload.Options, struct {
			Letter      string `json:"letter"`
			Description string `json:"description"`
		}{string(rune('A' + i)), opt.Description})
	}
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(payload); err != nil {
		return d, err
	}
	// Match Python json.dumps' default separators used by SemIf's direct_messages.
	compact := bytes.TrimSpace(buffer.Bytes())
	var expanded strings.Builder
	quoted, escaped := false, false
	for _, ch := range compact {
		if quoted {
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				quoted = false
			}
		} else if ch == '"' {
			quoted = true
		}
		expanded.WriteByte(ch)
		if !quoted && (ch == ':' || ch == ',') {
			expanded.WriteByte(' ')
		}
	}
	d.Prompt = expanded.String()
	return d, nil
}

type criterion struct {
	ID    string
	Value json.RawMessage
}

func orderedCriteria(raw json.RawMessage) ([]criterion, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	start, err := decoder.Token()
	if err != nil || start != json.Delim('{') {
		return nil, errors.New("choice criteria must be an object")
	}
	var options []criterion
	seen := map[string]bool{}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		id := key.(string)
		if seen[id] {
			return nil, fmt.Errorf("duplicate option ID %q", id)
		}
		seen[id] = true
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		options = append(options, criterion{id, value})
	}
	_, err = decoder.Token()
	return options, err
}

func (d decision) answer(values []float64) any {
	if d.Kind == "noul" {
		return map[string]any{"type": "noul", "noul": values[0]}
	}
	probabilities := make(map[string]float64, len(values))
	best := 0
	entropy := 0.0
	for i, p := range values {
		probabilities[d.Options[i].ID] = p
		if p > values[best] {
			best = i
		}
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	confidence := math.Max(0, math.Min(1, 1-entropy/math.Log(float64(len(values)))))
	if d.Kind == "choice" {
		return map[string]any{"type": "choice", "choice": d.Options[best].ID, "probabilities": probabilities, "confidence": confidence}
	}
	expected := 0.0
	for i, p := range values {
		expected += float64(i) * p
	}
	return map[string]any{"type": "score", "score": expected, "legend": d.Legend, "probabilities": probabilities, "confidence": confidence}
}
