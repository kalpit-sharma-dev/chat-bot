package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
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
	Message   string `json:"message"`
	SessionID string `json:"session_id,omitempty"`
	Token     string `json:"token,omitempty"`
}

type ChatResponse struct {
	Response  string `json:"response"`
	SessionID string `json:"session_id,omitempty"`
	Token     string `json:"token,omitempty"`
}

type AuthResponse struct {
	Token     string `json:"token"`
	SessionID string `json:"session_id"`
	ExpiresAt int64  `json:"expires_at"`
}

type StreamingSession struct {
	ID        string
	Content   string
	Done      bool
	LastPoll  time.Time
	Token     string
	CreatedAt time.Time
	ExpiresAt time.Time
	mu        sync.RWMutex
}

type UserSession struct {
	ID        string
	Token     string
	CreatedAt time.Time
	ExpiresAt time.Time
	LastUsed  time.Time
	mu        sync.RWMutex
}

var (
	streamingSessions   = make(map[string]*StreamingSession)
	streamingSessionsMu sync.RWMutex
	userSessions        = make(map[string]*UserSession)
	userSessionsMu      sync.RWMutex
	tokenToSession      = make(map[string]string)
	tokenToSessionMu    sync.RWMutex
)

const (
	TokenExpiry   = 24 * time.Hour
	SessionExpiry = 30 * time.Minute
	BufferSize    = 256 // Smaller buffer for faster streaming
)

func main() {
	http.HandleFunc("/auth", authHandler)
	http.HandleFunc("/chat", chatHandler)
	http.HandleFunc("/chat/stream", streamHandler)
	http.HandleFunc("/chat/poll/", pollHandler)
	http.HandleFunc("/health", healthHandler)

	// Clean up old sessions periodically
	go cleanupRoutine()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func generateToken() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func generateSessionID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func authHandler(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sessionID := generateSessionID()
	token := generateToken()
	now := time.Now()
	expiresAt := now.Add(TokenExpiry)

	session := &UserSession{
		ID:        sessionID,
		Token:     token,
		CreatedAt: now,
		ExpiresAt: expiresAt,
		LastUsed:  now,
	}

	userSessionsMu.Lock()
	userSessions[sessionID] = session
	userSessionsMu.Unlock()

	tokenToSessionMu.Lock()
	tokenToSession[token] = sessionID
	tokenToSessionMu.Unlock()

	log.Printf("Created new session: %s with token: %s", sessionID, token[:8]+"...")

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(AuthResponse{
		Token:     token,
		SessionID: sessionID,
		ExpiresAt: expiresAt.Unix(),
	})
}

func validateToken(token string) (*UserSession, bool) {
	if token == "" {
		return nil, false
	}

	tokenToSessionMu.RLock()
	sessionID, exists := tokenToSession[token]
	tokenToSessionMu.RUnlock()

	if !exists {
		return nil, false
	}

	userSessionsMu.RLock()
	session, exists := userSessions[sessionID]
	userSessionsMu.RUnlock()

	if !exists {
		return nil, false
	}

	session.mu.RLock()
	expired := time.Now().After(session.ExpiresAt)
	session.mu.RUnlock()

	if expired {
		// Clean up expired session
		cleanupSession(sessionID, token)
		return nil, false
	}

	// Update last used time
	session.mu.Lock()
	session.LastUsed = time.Now()
	session.mu.Unlock()

	return session, true
}

func cleanupSession(sessionID, token string) {
	userSessionsMu.Lock()
	delete(userSessions, sessionID)
	userSessionsMu.Unlock()

	tokenToSessionMu.Lock()
	delete(tokenToSession, token)
	tokenToSessionMu.Unlock()
}

func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization")
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("=== Chat request received ===")
	setCORSHeaders(w)

	if r.Method == http.MethodOptions {
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

	// Validate token
	session, valid := validateToken(req.Token)
	if !valid {
		log.Printf("Invalid or expired token: %s", req.Token[:8]+"...")
		http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
		return
	}

	log.Printf("Received message from session %s: %s", session.ID, req.Message)

	acceptHeader := r.Header.Get("Accept")
	log.Printf("Accept header: %s", acceptHeader)

	if acceptHeader == "text/event-stream" {
		handleStreamingResponse(w, req.Message, session)
	} else {
		handleNonStreamingResponse(w, req.Message, session)
	}
}

func handleStreamingResponse(w http.ResponseWriter, message string, session *UserSession) {
	log.Printf("Handling streaming response for session: %s", session.ID)

	// Set headers for Server-Sent Events (SSE)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Println("Streaming not supported")
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Send initial connection event
	fmt.Fprintf(w, "data: \n\n")
	flusher.Flush()

	chunkChan := make(chan string, BufferSize)
	errorChan := make(chan error, 1)

	go queryLlama3(message, chunkChan, errorChan)

	log.Println("Starting to stream responses to client")

	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				log.Println("Channel closed, ending stream")
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}

			// Send chunk immediately without buffering
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()

		case err := <-errorChan:
			log.Printf("Error from Llama3: %v", err)
			fmt.Fprintf(w, "data: Error: %s\n\n", err.Error())
			flusher.Flush()
			return

		case <-time.After(45 * time.Second):
			log.Println("Timeout waiting for response")
			fmt.Fprintf(w, "data: Request timeout\n\n")
			flusher.Flush()
			return
		}
	}
}

func handleNonStreamingResponse(w http.ResponseWriter, message string, session *UserSession) {
	log.Printf("Handling non-streaming response for session: %s", session.ID)

	w.Header().Set("Content-Type", "application/json")

	response, err := queryLlama3NonStreaming(message)
	if err != nil {
		log.Printf("Error getting response: %v", err)
		http.Error(w, fmt.Sprintf("Error: %v", err), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(ChatResponse{
		Response:  response,
		SessionID: session.ID,
	})
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

	llamaURL := "http://localhost:11434/api/generate"

	requestBody := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": 0.7,
			"top_p":       0.9,
			"top_k":       40,
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		errorChan <- fmt.Errorf("error preparing request")
		return
	}

	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	log.Println("Making request to Llama3")

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

		if done, ok := data["done"].(bool); ok && done {
			log.Println("Llama3 indicated completion")
			break
		}

		if chunk, ok := data["response"].(string); ok && chunk != "" {
			select {
			case chunkChan <- chunk:
				// Chunk sent successfully
			case <-time.After(1 * time.Second):
				log.Println("Timeout sending chunk to client")
				return
			}
		}
	}

	log.Println("Finished streaming from Llama3")
}

func queryLlama3NonStreaming(prompt string) (string, error) {
	log.Println("Starting queryLlama3NonStreaming")

	llamaURL := "http://localhost:11434/api/generate"

	requestBody := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"temperature": 0.7,
			"top_p":       0.9,
			"top_k":       40,
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	resp, err := client.Post(llamaURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("error calling Llama3: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Llama3 returned status: %d", resp.StatusCode)
	}

	var responseData map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&responseData); err != nil {
		return "", fmt.Errorf("error decoding response: %v", err)
	}

	if response, ok := responseData["response"].(string); ok {
		return response, nil
	}

	return "", fmt.Errorf("no response content found")
}

func streamHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("=== Stream request received ===")
	setCORSHeaders(w)

	if r.Method == http.MethodOptions {
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

	// Validate token
	userSession, valid := validateToken(req.Token)
	if !valid {
		log.Printf("Invalid or expired token")
		http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
		return
	}

	sessionID := fmt.Sprintf("%d-%s", time.Now().UnixNano(), userSession.ID)

	session := &StreamingSession{
		ID:        sessionID,
		Content:   "",
		Done:      false,
		LastPoll:  time.Now(),
		Token:     req.Token,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(SessionExpiry),
	}

	streamingSessionsMu.Lock()
	streamingSessions[sessionID] = session
	streamingSessionsMu.Unlock()

	log.Printf("Created streaming session %s for user session %s", sessionID, userSession.ID)

	go func() {
		defer func() {
			session.mu.Lock()
			session.Done = true
			session.mu.Unlock()
		}()

		queryLlama3Streaming(req.Message, session)
	}()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"session_id": sessionID})
}

func pollHandler(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/chat/poll/")
	sessionID := path

	// Get token from query params or header
	token := r.URL.Query().Get("token")
	if token == "" {
		token = r.Header.Get("Authorization")
		if strings.HasPrefix(token, "Bearer ") {
			token = strings.TrimPrefix(token, "Bearer ")
		}
	}

	// Validate token
	_, valid := validateToken(token)
	if !valid {
		http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
		return
	}

	streamingSessionsMu.RLock()
	session, exists := streamingSessions[sessionID]
	streamingSessionsMu.RUnlock()

	if !exists {
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	// Verify token matches session
	if session.Token != token {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
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

func queryLlama3Streaming(prompt string, session *StreamingSession) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from panic in queryLlama3Streaming: %v", r)
		}
	}()

	log.Printf("Starting queryLlama3Streaming for session: %s", session.ID)

	llamaURL := "http://localhost:11434/api/generate"

	requestBody := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": 0.7,
			"top_p":       0.9,
			"top_k":       40,
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		return
	}

	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	log.Println("Making request to Llama3")

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

		if done, ok := data["done"].(bool); ok && done {
			log.Println("Llama3 indicated completion")
			break
		}

		if chunk, ok := data["response"].(string); ok && chunk != "" {
			session.mu.Lock()
			session.Content += chunk
			session.mu.Unlock()

			log.Printf("Added chunk to session %s, total length: %d", session.ID, len(session.Content))
		}
	}

	log.Printf("Finished streaming from Llama3 for session %s", session.ID)
}

func cleanupRoutine() {
	ticker := time.NewTicker(2 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		now := time.Now()

		// Clean up expired user sessions
		userSessionsMu.Lock()
		for id, session := range userSessions {
			session.mu.RLock()
			expired := now.After(session.ExpiresAt)
			token := session.Token
			session.mu.RUnlock()

			if expired {
				delete(userSessions, id)

				tokenToSessionMu.Lock()
				delete(tokenToSession, token)
				tokenToSessionMu.Unlock()

				log.Printf("Cleaned up expired user session %s", id)
			}
		}
		userSessionsMu.Unlock()

		// Clean up expired streaming sessions
		streamingSessionsMu.Lock()
		for id, session := range streamingSessions {
			session.mu.RLock()
			lastPoll := session.LastPoll
			done := session.Done
			expired := now.After(session.ExpiresAt)
			session.mu.RUnlock()

			if (done && now.Sub(lastPoll) > 5*time.Minute) || expired {
				delete(streamingSessions, id)
				log.Printf("Cleaned up streaming session %s", id)
			}
		}
		streamingSessionsMu.Unlock()
	}
}
