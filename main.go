package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type ChatRequest struct {
	Message string `json:"message"`
}

type ChatResponse struct {
	Response string `json:"response"`
}

type StreamingSession struct {
	ID       string
	Content  string
	Done     bool
	LastPoll time.Time
	mu       sync.RWMutex
}

var (
	sessions   = make(map[string]*StreamingSession)
	sessionsMu sync.RWMutex
)

func main() {
	http.HandleFunc("/chat", chatHandler)
	http.HandleFunc("/chat/stream", streamHandler)
	http.HandleFunc("/chat/poll/", pollHandler)
	http.HandleFunc("/health", healthHandler)

	// Clean up old sessions periodically
	go cleanupSessions()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("=== Chat request received ===")

	// Handle CORS preflight requests
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept")
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse the request body
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Error decoding request: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	log.Printf("Received message: %s", req.Message)

	// Check Accept header to determine response type
	acceptHeader := r.Header.Get("Accept")
	log.Printf("Accept header: %s", acceptHeader)

	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept")

	// Handle streaming vs non-streaming based on Accept header
	if acceptHeader == "text/event-stream" {
		handleStreamingResponse(w, req.Message)
	} else {
		handleNonStreamingResponse(w, req.Message)
	}
}

func handleStreamingResponse(w http.ResponseWriter, message string) {
	log.Println("Handling streaming response")

	// Set headers for Server-Sent Events (SSE)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Create a flusher to push chunks to client
	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Println("Streaming not supported")
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Create a channel to stream chunks
	chunkChan := make(chan string, 100) // Buffered channel
	errorChan := make(chan error, 1)

	// Start streaming from Llama3 in a goroutine
	go queryLlama3(message, chunkChan, errorChan)

	log.Println("Starting to stream responses to client")

	// Stream responses to client
	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				log.Println("Channel closed, ending stream")
				return
			}

			// Format as SSE event and send immediately
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			fmt.Println(chunk)
			flusher.Flush()

		case err := <-errorChan:
			log.Printf("Error from Llama3: %v", err)
			fmt.Fprintf(w, "data: Error: %s\n\n", err.Error())
			flusher.Flush()
			return

		case <-time.After(30 * time.Second):
			log.Println("Timeout waiting for response")
			fmt.Fprintf(w, "data: Request timeout\n\n")
			flusher.Flush()
			return
		}
	}
}

func handleNonStreamingResponse(w http.ResponseWriter, message string) {
	log.Println("Handling non-streaming response")

	w.Header().Set("Content-Type", "text/plain")

	// Get complete response from Llama3
	response, err := queryLlama3NonStreaming(message)
	if err != nil {
		log.Printf("Error getting response: %v", err)
		http.Error(w, fmt.Sprintf("Error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Write([]byte(response))
}

func queryLlama3(prompt string, chunkChan chan<- string, errorChan chan<- error) {
	defer close(chunkChan)

	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from panic in queryLlama3: %v", r)
			errorChan <- fmt.Errorf("internal server error")
		}
	}()

	log.Println("Starting queryLlama3")

	// Docker container URL where Llama3 is running
	llamaURL := "http://localhost:11434/api/generate"

	// Prepare the request to Llama3
	requestBody := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": true,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		errorChan <- fmt.Errorf("error preparing request")
		return
	}

	// Create HTTP client with longer timeout
	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	log.Println("Making request to Llama3")

	// Make request to Llama3
	resp, err := client.Post(llamaURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		log.Printf("Error calling Llama3: %v", err)
		errorChan <- fmt.Errorf("error connecting to AI service")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Llama3 returned status: %d", resp.StatusCode)
		errorChan <- fmt.Errorf("AI service unavailable (status: %d)", resp.StatusCode)
		return
	}

	log.Println("Starting to stream response from Llama3")

	// Stream the response from Llama3
	decoder := json.NewDecoder(resp.Body)

	for {
		var data map[string]interface{}
		if err := decoder.Decode(&data); err != nil {
			if err == io.EOF {
				log.Println("Reached end of Llama3 response")
				break
			}
			log.Printf("Error decoding response: %v", err)
			errorChan <- fmt.Errorf("error processing response")
			return
		}

		// Check if this is the final response
		if done, ok := data["done"].(bool); ok && done {
			log.Println("Llama3 indicated completion")
			break
		}

		// Extract the response chunk
		if chunk, ok := data["response"].(string); ok && chunk != "" {
			log.Printf("Sending chunk: %q", chunk)

			// Send chunk to client
			select {
			case chunkChan <- chunk:
				// Chunk sent successfully
			case <-time.After(5 * time.Second):
				log.Println("Timeout sending chunk to client")
				return
			}
		}
	}

	log.Println("Finished streaming from Llama3")
}

func queryLlama3NonStreaming(prompt string) (string, error) {
	log.Println("Starting queryLlama3NonStreaming")

	// Docker container URL where Llama3 is running
	llamaURL := "http://localhost:11434/api/generate"

	// Prepare the request to Llama3 (non-streaming)
	requestBody := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": false,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	// Create HTTP client
	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	// Make request to Llama3
	resp, err := client.Post(llamaURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("error calling Llama3: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Llama3 returned status: %d", resp.StatusCode)
	}

	// Read the complete response
	var responseData map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&responseData); err != nil {
		return "", fmt.Errorf("error decoding response: %v", err)
	}

	if response, ok := responseData["response"].(string); ok {
		fmt.Println(response)
		return response, nil
	}

	return "", fmt.Errorf("no response content found")
}

func streamHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("=== Stream request received ===")

	// Handle CORS
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Error decoding request: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Generate session ID (this should match the assistantMessageId from frontend)
	sessionID := fmt.Sprintf("%d-ai", time.Now().UnixNano())

	// Create streaming session
	session := &StreamingSession{
		ID:       sessionID,
		Content:  "",
		Done:     false,
		LastPoll: time.Now(),
	}

	sessionsMu.Lock()
	sessions[sessionID] = session
	sessionsMu.Unlock()

	log.Printf("Created session %s for message: %s", sessionID, req.Message)

	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Start streaming in background
	go func() {
		defer func() {
			session.mu.Lock()
			session.Done = true
			session.mu.Unlock()
		}()

		queryLlama3Streaming(req.Message, session)
	}()

	// Return session ID
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"session_id": sessionID})
}

func pollHandler(w http.ResponseWriter, r *http.Request) {
	// Handle CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract session ID from URL
	path := strings.TrimPrefix(r.URL.Path, "/chat/poll/")
	sessionID := path

	sessionsMu.RLock()
	session, exists := sessions[sessionID]
	sessionsMu.RUnlock()

	if !exists {
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	session.mu.Lock()
	content := session.Content
	done := session.Done
	session.LastPoll = time.Now()
	session.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"content": content,
		"done":    done,
	})
}

func cleanupSessions() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		sessionsMu.Lock()
		for id, session := range sessions {
			session.mu.RLock()
			lastPoll := session.LastPoll
			done := session.Done
			session.mu.RUnlock()

			// Remove sessions that are done and haven't been polled for 5 minutes
			if done && time.Since(lastPoll) > 5*time.Minute {
				delete(sessions, id)
				log.Printf("Cleaned up session %s", id)
			}
		}
		sessionsMu.Unlock()
	}
}

func queryLlama3Streaming(prompt string, session *StreamingSession) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from panic in queryLlama3Streaming: %v", r)
		}
	}()

	log.Println("Starting queryLlama3Streaming")

	// Docker container URL where Llama3 is running
	llamaURL := "http://localhost:11434/api/generate"

	// Prepare the request to Llama3
	requestBody := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": true,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		return
	}

	// Create HTTP client
	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	log.Println("Making request to Llama3")

	// Make request to Llama3
	resp, err := client.Post(llamaURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		log.Printf("Error calling Llama3: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Llama3 returned status: %d", resp.StatusCode)
		return
	}

	log.Println("Starting to stream response from Llama3")

	// Stream the response from Llama3
	decoder := json.NewDecoder(resp.Body)

	for {
		var data map[string]interface{}
		if err := decoder.Decode(&data); err != nil {
			if err == io.EOF {
				log.Println("Reached end of Llama3 response")
				break
			}
			log.Printf("Error decoding response: %v", err)
			return
		}

		// Check if this is the final response
		if done, ok := data["done"].(bool); ok && done {
			log.Println("Llama3 indicated completion")
			break
		}

		// Extract the response chunk
		if chunk, ok := data["response"].(string); ok && chunk != "" {
			session.mu.Lock()
			session.Content += chunk
			session.mu.Unlock()

			log.Printf("Added chunk to session %s, total length: %d", session.ID, len(session.Content))
		}
	}

	log.Printf("Finished streaming from Llama3 for session %s", session.ID)
}
