package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	maxMediaItems = 8
	maxMediaBytes = 32 << 20
	maxMediaTotal = 48 << 20
)

type mediaState struct {
	Evidence json.RawMessage
	Data     []string
	Marker   string
}

func isMultimodal(state json.RawMessage) bool {
	var header struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(state, &header) == nil && header.Type == "multimodal"
}

func parseMediaState(raw json.RawMessage, props serverProps) (mediaState, error) {
	var state struct {
		Type    string            `json:"type"`
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &state); err != nil || state.Type != "multimodal" || len(state.Content) == 0 {
		return mediaState{}, errors.New("multimodal state requires a nonempty content array")
	}
	if props.MediaMarker == "" {
		return mediaState{}, errors.New("llama-server did not supply a media marker")
	}
	result := mediaState{Marker: props.MediaMarker}
	var evidence strings.Builder
	mediaBytes := 0
	for index, rawPart := range state.Content {
		var part struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			ImageURL struct {
				URL string `json:"url"`
			} `json:"image_url"`
			InputAudio struct {
				Data string `json:"data"`
				URL  string `json:"url"`
			} `json:"input_audio"`
			InputVideo struct {
				Data string `json:"data"`
				URL  string `json:"url"`
			} `json:"input_video"`
		}
		if err := json.Unmarshal(rawPart, &part); err != nil || part.Type == "" {
			return mediaState{}, fmt.Errorf("content[%d] must be a typed object", index)
		}
		if part.Type == "text" {
			if strings.TrimSpace(part.Text) == "" || strings.Contains(part.Text, props.MediaMarker) {
				return mediaState{}, fmt.Errorf("content[%d] must contain nonempty text without the server media marker", index)
			}
			evidence.WriteString(part.Text)
			evidence.WriteByte('\n')
			continue
		}
		if len(result.Data) >= maxMediaItems {
			return mediaState{}, fmt.Errorf("multimodal state permits at most %d media parts", maxMediaItems)
		}
		var source, modality, label string
		switch part.Type {
		case "image_url":
			source, modality, label = part.ImageURL.URL, "vision", "Image"
		case "input_audio":
			if part.InputAudio.Data != "" && part.InputAudio.URL != "" {
				return mediaState{}, fmt.Errorf("content[%d] must specify only one audio source", index)
			}
			source, modality, label = part.InputAudio.Data+part.InputAudio.URL, "audio", "Audio"
		case "input_video":
			if part.InputVideo.Data != "" && part.InputVideo.URL != "" {
				return mediaState{}, fmt.Errorf("content[%d] must specify only one video source", index)
			}
			source, modality, label = part.InputVideo.Data+part.InputVideo.URL, "video", "Video"
		default:
			return mediaState{}, fmt.Errorf("content[%d] has unsupported type %q", index, part.Type)
		}
		if !props.Modalities[modality] {
			return mediaState{}, fmt.Errorf("llama-server's loaded model does not support %s input", modality)
		}
		data, size, err := inlineMedia(source, modality)
		if err != nil {
			return mediaState{}, fmt.Errorf("content[%d]: %w", index, err)
		}
		mediaBytes += size
		if mediaBytes > maxMediaTotal {
			return mediaState{}, fmt.Errorf("combined media exceeds %d bytes", maxMediaTotal)
		}
		result.Data = append(result.Data, data)
		fmt.Fprintf(&evidence, "%s %d: %s\n", label, len(result.Data), props.MediaMarker)
	}
	if len(result.Data) == 0 {
		return mediaState{}, errors.New("multimodal state requires at least one media part")
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(evidence.String()); err != nil {
		return mediaState{}, err
	}
	result.Evidence = bytes.TrimSpace(encoded.Bytes())
	return result, nil
}

func inlineMedia(source, modality string) (string, int, error) {
	if strings.HasPrefix(source, "data:") {
		metadata, payload, ok := strings.Cut(source[5:], ",")
		if !ok || !strings.HasSuffix(strings.ToLower(metadata), ";base64") || !strings.HasPrefix(strings.ToLower(metadata), mediaMIMEPrefix(modality)) {
			return "", 0, fmt.Errorf("expected a base64 data:%s... URL", mediaMIMEPrefix(modality))
		}
		source = payload
	} else if modality == "vision" || strings.Contains(source, "://") {
		return "", 0, errors.New("use an inline base64 data URL; remote and file URLs are not supported")
	}
	if source == "" || base64.StdEncoding.DecodedLen(len(source)) > maxMediaBytes+2 {
		return "", 0, fmt.Errorf("media must contain at most %d decoded bytes", maxMediaBytes)
	}
	decoded, err := base64.StdEncoding.Strict().DecodeString(source)
	if err != nil {
		decoded, err = base64.RawStdEncoding.Strict().DecodeString(source)
	}
	if err != nil || len(decoded) == 0 || len(decoded) > maxMediaBytes {
		return "", 0, errors.New("media must be nonempty, valid base64 within the size limit")
	}
	return base64.StdEncoding.EncodeToString(decoded), len(decoded), nil
}

func mediaMIMEPrefix(modality string) string {
	switch modality {
	case "vision":
		return "image/"
	case "audio":
		return "audio/"
	default:
		return "video/"
	}
}
