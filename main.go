package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"time"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8090", "System One HTTP listen address")
	upstream := flag.String("llama-url", "http://127.0.0.1:8080", "llama-server base URL")
	model := flag.String("model", "semif-local", "model name returned by this service")
	initial := flag.Int("initial-top-probs", 256, "initial number of next-token log-probabilities to request")
	maximum := flag.Int("max-top-probs", 262144, "largest next-token top-N request allowed")
	maxPrompt := flag.Int("max-prompt-tokens", 4096, "maximum prompt tokens per question")
	flag.Parse()
	if *initial < 1 || *maximum < *initial || *maxPrompt < 1 {
		log.Fatal("initial-top-probs, max-top-probs, and max-prompt-tokens must be positive; maximum must be >= initial")
	}
	backend, err := newBackend(*upstream, os.Getenv("LLAMA_API_KEY"), *initial, *maximum, *maxPrompt)
	if err != nil {
		log.Fatal(err)
	}
	service := &service{backend: backend, model: *model, apiKey: os.Getenv("SEMIF_API_KEY")}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/systemone", service.evaluate)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	server := &http.Server{Addr: *listen, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Printf("System One adapter listening on %s (llama-server %s)", *listen, *upstream)
	log.Fatal(server.ListenAndServe())
}
