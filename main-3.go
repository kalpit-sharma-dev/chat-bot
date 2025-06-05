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
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ===== Core Types (keeping your existing types) =====
type ChatRequest struct {
	Message   string `json:"message"`
	SessionID string `json:"session_id,omitempty"`
	Token     string `json:"token,omitempty"`
}

type ChatResponse struct {
	Response  string   `json:"response"`
	SessionID string   `json:"session_id,omitempty"`
	Token     string   `json:"token,omitempty"`
	Intent    string   `json:"intent,omitempty"`
	Tools     []string `json:"tools,omitempty"`
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
	ID          string
	Token       string
	CreatedAt   time.Time
	ExpiresAt   time.Time
	LastUsed    time.Time
	AccountID   string // Link to user's bank account
	CurrentStep *ConversationStep
	mu          sync.RWMutex
}

// ===== New Banking Types =====
type ConversationMessage struct {
	ID        string            `json:"id"`
	Role      string            `json:"role"` // "user" or "assistant"
	Content   string            `json:"content"`
	Timestamp time.Time         `json:"timestamp"`
	Intent    string            `json:"intent,omitempty"`
	Tools     []string          `json:"tools,omitempty"`
	Entities  map[string]string `json:"entities,omitempty"`
}

type ConversationHistory struct {
	SessionID string                 `json:"session_id"`
	Messages  []ConversationMessage  `json:"messages"`
	Context   map[string]interface{} `json:"context"`
	mu        sync.RWMutex
}

type Intent struct {
	Name       string                 `json:"name"`
	Confidence float64                `json:"confidence"`
	Entities   map[string]string      `json:"entities"`
	Parameters map[string]interface{} `json:"parameters"`
}

type IntentPattern struct {
	Name           string   `json:"name"`
	Patterns       []string `json:"patterns"`
	Keywords       []string `json:"keywords"`
	Action         string   `json:"action"`
	RequiredParams []string `json:"required_params"`
}

type ToolResult struct {
	Success  bool        `json:"success"`
	Data     interface{} `json:"data,omitempty"`
	Error    string      `json:"error,omitempty"`
	Message  string      `json:"message,omitempty"`
	NextStep string      `json:"next_step,omitempty"`
}

type BankingFunction struct {
	Name           string                 `json:"name"`
	Description    string                 `json:"description"`
	Parameters     map[string]interface{} `json:"parameters"`
	Handler        func(params map[string]interface{}) ToolResult
	RequiredParams []string `json:"required_params"`
}

type Account struct {
	AccountNumber string    `json:"account_number"`
	AccountType   string    `json:"account_type"`
	Balance       float64   `json:"balance"`
	Currency      string    `json:"currency"`
	LastUpdated   time.Time `json:"last_updated"`
}

type Payee struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	AccountNo string    `json:"account_no"`
	BankName  string    `json:"bank_name"`
	IFSCCode  string    `json:"ifsc_code"`
	PayeeType string    `json:"payee_type"`
	UPIId     string    `json:"upi_id,omitempty"`
	AddedDate time.Time `json:"added_date"`
	IsActive  bool      `json:"is_active"`
}

type TransferRequest struct {
	FromAccount string    `json:"from_account"`
	ToAccount   string    `json:"to_account"`
	Amount      float64   `json:"amount"`
	Method      string    `json:"method"`
	Description string    `json:"description"`
	TransferID  string    `json:"transfer_id"`
	Status      string    `json:"status"`
	Timestamp   time.Time `json:"timestamp"`
}

type ConversationStep struct {
	StepID     string                 `json:"step_id"`
	Intent     string                 `json:"intent"`
	Parameters map[string]interface{} `json:"parameters"`
	Missing    []string               `json:"missing"`
	Question   string                 `json:"question"`
	Complete   bool                   `json:"complete"`
}

type WeatherData struct {
	Location    string    `json:"location"`
	Temperature float64   `json:"temperature"`
	Condition   string    `json:"condition"`
	Humidity    int       `json:"humidity"`
	WindSpeed   string    `json:"wind_speed"`
	LastUpdated time.Time `json:"last_updated"`
}

// ===== Llama3 Integration Types =====
type LlamaRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options"`
	System  string                 `json:"system,omitempty"`
}

type LlamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// ===== Global State =====
var (
	streamingSessions   = make(map[string]*StreamingSession)
	streamingSessionsMu sync.RWMutex
	userSessions        = make(map[string]*UserSession)
	userSessionsMu      sync.RWMutex
	tokenToSession      = make(map[string]string)
	tokenToSessionMu    sync.RWMutex

	// New banking state
	conversationHistory = make(map[string]*ConversationHistory)
	conversationMu      sync.RWMutex
	bankingFunctions    = make(map[string]BankingFunction)
	intentPatterns      []IntentPattern
	userAccounts        = make(map[string][]Account)
	userPayees          = make(map[string][]Payee)
	transferHistory     = make(map[string][]TransferRequest)
	weatherCache        = make(map[string]WeatherData)
	weatherCacheMu      sync.RWMutex
)

const (
	TokenExpiry        = 24 * time.Hour
	SessionExpiry      = 30 * time.Minute
	BufferSize         = 256
	WeatherCacheExpiry = 15 * time.Minute
)

// // System prompt for banking context
// const bankingSystemPrompt = `You are an AI banking assistant with access to various banking functions.
// You help users with fund transfers, balance inquiries, deposits, payee management, weather information, and transaction history.
// Always be helpful, accurate, and security-conscious. When users request banking operations, explain what you're doing.
// Available functions: fund_transfer, add_payee, view_balance, create_fd, create_rd, get_interest_rates, get_weather, get_payees, transfer_history.`

const bankingSystemPrompt = `You are a secure and intelligent AI banking assistant integrated into a digital banking system.

Your role is to help users perform a wide range of banking tasks safely, efficiently, and clearly. Always ensure user intent is well-understood, confirm sensitive operations, and provide helpful, accurate guidance at every step.

You have access to the following banking functions:
- fund_transfer: Transfer funds to a saved payee. Confirm the recipient name and amount before initiating the transaction.
- add_payee: Add a new payee with details like name, account number, and IFSC. Ensure confirmation before saving.
- view_balance: Provide the current account balance on request.
- create_fd: Create a fixed deposit by specifying amount and duration.
- create_rd: Create a recurring deposit with monthly contributions and term.
- get_interest_rates: Fetch the latest interest rates for FD and RD products.
- get_weather: Provide current weather details for a specified location.
- get_payees: Retrieve the list of all saved payees.
- transfer_history: Display recent transactions or money transfers.

Guidelines:
- Be professional, concise, and user-friendly in all responses.
- Always maintain security and confidentiality. Never reveal or assume sensitive information unless explicitly provided.
- Confirm high-risk operations like fund_transfer or add_payee before execution.
- Explain the purpose of each action you're about to take, especially for financial operations.
- If a user seems unsure, guide them step-by-step.

You are here to make banking simpler, safer, and smarter for the user.`

func main() {
	// Initialize banking functions and data
	initializeBankingFunctions()
	initializeIntentPatterns()
	initializeMockData()

	// Keep your existing handlers
	http.HandleFunc("/auth", authHandler)
	http.HandleFunc("/chat", enhancedChatHandler) // Enhanced version
	http.HandleFunc("/chat/stream", streamHandler)
	http.HandleFunc("/chat/poll/", pollHandler)
	http.HandleFunc("/health", healthHandler)

	// Add new banking endpoints
	http.HandleFunc("/conversation/history/", conversationHistoryHandler)

	// Clean up old sessions periodically
	go cleanupRoutine()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Enhanced Banking Server with Llama3 starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// ===== Keep all your existing functions (generateToken, generateSessionID, etc.) =====
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

func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization")
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
		cleanupSession(sessionID, token)
		return nil, false
	}

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

// ===== Enhanced Auth Handler with Banking Account Link =====
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
		AccountID: "user123", // Mock user account
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

// ===== Enhanced Chat Handler with Banking Context =====
func enhancedChatHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("=== Enhanced chat request received ===")
	setCORSHeaders(w)
	fmt.Println("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", r.Method)
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

	// Get or create conversation history
	conversation := getOrCreateConversation(session.ID)

	// Detect intent
	intent := recognizeIntent(req.Message)

	// Add user message to conversation
	conversation.AddMessage("user", req.Message, intent.Name, nil, intent.Entities)

	log.Printf("Received message from session %s: %s (Intent: %s)", session.ID, req.Message, intent.Name)

	acceptHeader := r.Header.Get("Accept")
	log.Printf("Accept header: %s", acceptHeader)

	if acceptHeader == "text/event-stream" {
		handleEnhancedStreamingResponse(w, req.Message, session, conversation, intent)
	} else {
		handleEnhancedNonStreamingResponse(w, req.Message, session, conversation, intent)
	}
}

// ===== Enhanced Streaming Response with Banking Context =====
func handleEnhancedStreamingResponse(w http.ResponseWriter, message string, session *UserSession, conversation *ConversationHistory, intent Intent) {
	log.Printf("Handling enhanced streaming response for session: %s", session.ID)

	// Set headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

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
	toolResultChan := make(chan ToolResult, 1)

	// Process with banking context
	go processMessageWithBanking(message, session, conversation, intent, chunkChan, errorChan, toolResultChan)

	var fullResponse strings.Builder
	var usedTools []string

	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				// Add assistant message to conversation
				conversation.AddMessage("assistant", fullResponse.String(), intent.Name, usedTools, nil)

				log.Println("Channel closed, ending stream")
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}

			fullResponse.WriteString(chunk)
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()

		case toolResult := <-toolResultChan:
			if toolResult.Success {
				usedTools = append(usedTools, intent.Name)
				// Send tool result as a special event
				toolData, _ := json.Marshal(map[string]interface{}{
					"type": "tool_result",
					"data": toolResult.Data,
				})
				fmt.Fprintf(w, "event: tool\ndata: %s\n\n", toolData)
				flusher.Flush()
			}

		case err := <-errorChan:
			log.Printf("Error from processing: %v", err)
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

// ===== Process Message with Banking Context =====
func processMessageWithBanking(message string, session *UserSession, conversation *ConversationHistory, intent Intent, chunkChan chan<- string, errorChan chan<- error, toolResultChan chan<- ToolResult) {
	defer close(chunkChan)

	// Check if this is a continuation of a multi-step conversation
	if session.CurrentStep != nil && !session.CurrentStep.Complete {
		handleMultiStepStreaming(message, intent, session, conversation, chunkChan, toolResultChan)
		return
	}

	// If high confidence banking intent, execute function first
	if intent.Confidence > 0.7 && intent.Name != "general" {
		if function, exists := bankingFunctions[intent.Name]; exists {
			params := extractParametersFromMessage(message, intent, conversation)
			missingParams := checkMissingParameters(function, params)

			if len(missingParams) == 0 {
				// Execute banking function
				result := function.Handler(params)
				toolResultChan <- result

				// Generate natural response about the action
				prompt := fmt.Sprintf("%s\n\nUser request: %s\n\nYou just executed a banking function with this result: %s\n\nProvide a natural, friendly response explaining what was done.",
					bankingSystemPrompt, message, result.Message)

				queryLlama3WithContext(prompt, conversation.GetContext(), chunkChan, errorChan)
				return
			} else {
				// Start multi-step conversation
				session.CurrentStep = &ConversationStep{
					StepID:     fmt.Sprintf("step_%d", time.Now().UnixNano()),
					Intent:     intent.Name,
					Parameters: params,
					Missing:    missingParams,
					Complete:   false,
				}

				question := askForMissingParameters(missingParams, intent.Name)
				for _, char := range question {
					chunkChan <- string(char)
					time.Sleep(10 * time.Millisecond)
				}
				return
			}
		}
	}

	// For general queries or low confidence, use Llama3 with banking context
	context := conversation.GetContext()
	enhancedPrompt := fmt.Sprintf("%s\n\nConversation context:\n%s\n\nUser: %s\n\nAssistant:",
		bankingSystemPrompt, context, message)

	queryLlama3WithContext(enhancedPrompt, context, chunkChan, errorChan)
}

// ===== Llama3 Query with Context =====
func queryLlama3WithContext(prompt string, context string, chunkChan chan<- string, errorChan chan<- error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from panic in queryLlama3WithContext: %v", r)
			errorChan <- fmt.Errorf("internal server error")
		}
	}()

	llamaURL := "http://localhost:11434/api/generate"

	requestBody := LlamaRequest{
		Model:  "llama3",
		Prompt: prompt,
		Stream: true,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"top_p":       0.9,
			"top_k":       40,
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		errorChan <- fmt.Errorf("error preparing request")
		return
	}

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Post(llamaURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		errorChan <- fmt.Errorf("error connecting to AI service")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errorChan <- fmt.Errorf("AI service unavailable")
		return
	}

	decoder := json.NewDecoder(resp.Body)
	for {
		var data LlamaResponse
		if err := decoder.Decode(&data); err != nil {
			if err == io.EOF {
				break
			}
			errorChan <- fmt.Errorf("error processing response")
			return
		}

		if data.Done {
			break
		}

		if data.Response != "" {
			select {
			case chunkChan <- data.Response:
			case <-time.After(1 * time.Second):
				return
			}
		}
	}
}

// ===== Banking Functions Implementation =====
func initializeBankingFunctions() {
	bankingFunctions["fund_transfer"] = BankingFunction{
		Name:           "fund_transfer",
		Description:    "Transfer funds using UPI, IMPS, or NEFT",
		RequiredParams: []string{"amount", "method"},
		Handler:        handleFundTransfer,
	}

	bankingFunctions["view_balance"] = BankingFunction{
		Name:           "view_balance",
		Description:    "View account balance",
		RequiredParams: []string{},
		Handler:        handleViewBalance,
	}

	bankingFunctions["get_weather"] = BankingFunction{
		Name:           "get_weather",
		Description:    "Get current weather information",
		RequiredParams: []string{},
		Handler:        handleGetWeather,
	}

	// Add more banking functions as needed...
}

func initializeIntentPatterns() {
	intentPatterns = []IntentPattern{
		{
			Name:     "fund_transfer",
			Patterns: []string{"transfer money", "send money", "pay", "transfer funds"},
			Keywords: []string{"transfer", "send", "pay", "money", "funds", "upi", "imps", "neft"},
			Action:   "fund_transfer",
		},
		{
			Name:     "check_balance",
			Patterns: []string{"check balance", "account balance", "show balance"},
			Keywords: []string{"balance", "account", "check", "show"},
			Action:   "view_balance",
		},
		{
			Name:     "weather",
			Patterns: []string{"weather", "temperature", "forecast"},
			Keywords: []string{"weather", "temperature", "forecast", "climate"},
			Action:   "get_weather",
		},
	}
}

func initializeMockData() {
	// Initialize mock user accounts
	userAccounts["user123"] = []Account{
		{
			AccountNumber: "1234567890",
			AccountType:   "Savings",
			Balance:       150000.00,
			Currency:      "INR",
			LastUpdated:   time.Now(),
		},
		{
			AccountNumber: "0987654321",
			AccountType:   "Current",
			Balance:       250000.00,
			Currency:      "INR",
			LastUpdated:   time.Now(),
		},
	}
}

// ===== Intent Recognition =====
func recognizeIntent(message string) Intent {
	message = strings.ToLower(message)
	bestIntent := Intent{Name: "general", Confidence: 0.0, Entities: make(map[string]string)}

	for _, pattern := range intentPatterns {
		confidence := 0.0

		// Check patterns
		for _, p := range pattern.Patterns {
			if strings.Contains(message, strings.ToLower(p)) {
				confidence += 0.8
				break
			}
		}

		// Check keywords
		keywordMatches := 0
		for _, keyword := range pattern.Keywords {
			if strings.Contains(message, keyword) {
				keywordMatches++
			}
		}

		if len(pattern.Keywords) > 0 {
			confidence += float64(keywordMatches) / float64(len(pattern.Keywords)) * 0.4
		}

		if confidence > bestIntent.Confidence {
			bestIntent = Intent{
				Name:       pattern.Name,
				Confidence: confidence,
				Entities:   extractEntities(message, pattern.Name),
				Parameters: make(map[string]interface{}),
			}
		}
	}

	return bestIntent
}

func extractEntities(message string, intentName string) map[string]string {
	entities := make(map[string]string)

	// Extract amounts
	amountRegex := regexp.MustCompile(`(?:₹|rs\.?|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs|rupees|inr)?`)
	if matches := amountRegex.FindStringSubmatch(message); len(matches) > 1 {
		entities["amount"] = strings.ReplaceAll(matches[1], ",", "")
	}

	// Extract locations for weather
	if intentName == "weather" {
		locationRegex := regexp.MustCompile(`(?:in|at|for)\s+([a-zA-Z\s]+)`)
		if matches := locationRegex.FindStringSubmatch(message); len(matches) > 1 {
			entities["location"] = strings.TrimSpace(matches[1])
		}
	}

	return entities
}

// ===== Conversation Management =====
func getOrCreateConversation(sessionID string) *ConversationHistory {
	conversationMu.Lock()
	defer conversationMu.Unlock()

	if conv, exists := conversationHistory[sessionID]; exists {
		return conv
	}

	conv := &ConversationHistory{
		SessionID: sessionID,
		Messages:  []ConversationMessage{},
		Context:   make(map[string]interface{}),
	}
	conversationHistory[sessionID] = conv
	return conv
}

func (ch *ConversationHistory) AddMessage(role, content, intent string, tools []string, entities map[string]string) {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	message := ConversationMessage{
		ID:        fmt.Sprintf("%d", time.Now().UnixNano()),
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Intent:    intent,
		Tools:     tools,
		Entities:  entities,
	}

	ch.Messages = append(ch.Messages, message)

	// Keep only last 20 messages
	if len(ch.Messages) > 20 {
		ch.Messages = ch.Messages[len(ch.Messages)-20:]
	}
}

func (ch *ConversationHistory) GetContext() string {
	ch.mu.RLock()
	defer ch.mu.RUnlock()

	if len(ch.Messages) == 0 {
		return ""
	}

	context := ""
	start := len(ch.Messages) - 5
	if start < 0 {
		start = 0
	}

	for _, msg := range ch.Messages[start:] {
		context += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}

	return context
}

// ===== Sample Banking Functions =====
func handleFundTransfer(params map[string]interface{}) ToolResult {
	amountStr, _ := params["amount"].(string)
	method, _ := params["method"].(string)

	amount, err := strconv.ParseFloat(amountStr, 64)
	if err != nil {
		return ToolResult{Success: false, Error: "Invalid amount"}
	}

	// Mock transfer
	transferID := fmt.Sprintf("TXN%d", time.Now().UnixNano())

	return ToolResult{
		Success: true,
		Data: map[string]interface{}{
			"transfer_id": transferID,
			"amount":      amount,
			"method":      method,
			"status":      "SUCCESS",
		},
		Message: fmt.Sprintf("Transfer of ₹%.2f via %s completed successfully. Reference: %s", amount, method, transferID),
	}
}

func handleViewBalance(params map[string]interface{}) ToolResult {
	accounts := userAccounts["user123"]

	var balanceInfo []map[string]interface{}
	totalBalance := 0.0

	for _, account := range accounts {
		balanceInfo = append(balanceInfo, map[string]interface{}{
			"account_number": account.AccountNumber,
			"account_type":   account.AccountType,
			"balance":        account.Balance,
		})
		totalBalance += account.Balance
	}

	return ToolResult{
		Success: true,
		Data:    balanceInfo,
		Message: fmt.Sprintf("Total balance across all accounts: ₹%.2f", totalBalance),
	}
}

func handleGetWeather(params map[string]interface{}) ToolResult {
	location, _ := params["location"].(string)
	if location == "" {
		location = "Delhi"
	}

	// Mock weather data
	weather := WeatherData{
		Location:    location,
		Temperature: 25.0 + float64(time.Now().Hour()%10),
		Condition:   "Partly Cloudy",
		Humidity:    65,
		WindSpeed:   "12 km/h",
		LastUpdated: time.Now(),
	}

	return ToolResult{
		Success: true,
		Data:    weather,
		Message: fmt.Sprintf("Weather in %s: %.1f°C, %s", location, weather.Temperature, weather.Condition),
	}
}

// ===== Helper Functions =====
func extractParametersFromMessage(message string, intent Intent, conversation *ConversationHistory) map[string]interface{} {
	params := make(map[string]interface{})

	for key, value := range intent.Entities {
		params[key] = value
	}

	// Add default parameters based on intent
	switch intent.Name {
	case "fund_transfer":
		if _, exists := params["method"]; !exists {
			if strings.Contains(strings.ToLower(message), "upi") {
				params["method"] = "UPI"
			} else {
				params["method"] = "NEFT"
			}
		}
	}

	return params
}

func checkMissingParameters(function BankingFunction, params map[string]interface{}) []string {
	var missing []string
	for _, required := range function.RequiredParams {
		if _, exists := params[required]; !exists {
			missing = append(missing, required)
		}
	}
	return missing
}

// ===== Complete the askForMissingParameters function =====
func askForMissingParameters(missing []string, intentName string) string {
	if len(missing) == 0 {
		return ""
	}

	param := missing[0] // Ask for one parameter at a time

	switch intentName {
	case "fund_transfer":
		switch param {
		case "amount":
			return "How much would you like to transfer?"
		case "method":
			return "Which transfer method would you prefer: UPI, IMPS, or NEFT?"
		case "to_account":
			return "What's the recipient's account number or UPI ID?"
		}
	case "get_weather":
		switch param {
		case "location":
			return "Which city would you like to know the weather for?"
		}
	}

	return fmt.Sprintf("I need more information. Could you provide the %s?", param)
}

// ===== Handle Multi-Step Streaming =====
func handleMultiStepStreaming(message string, intent Intent, session *UserSession, conversation *ConversationHistory, chunkChan chan<- string, toolResultChan chan<- ToolResult) {
	step := session.CurrentStep

	// Extract parameters from the response
	for _, missing := range step.Missing {
		if value, exists := intent.Entities[missing]; exists {
			step.Parameters[missing] = value
		} else {
			// Try to extract from message directly
			if missing == "amount" {
				amountRegex := regexp.MustCompile(`\d+(?:\.\d+)?`)
				if matches := amountRegex.FindStringSubmatch(message); len(matches) > 0 {
					step.Parameters[missing] = matches[0]
				}
			}
		}
	}

	// Update missing parameters
	if function, exists := bankingFunctions[step.Intent]; exists {
		step.Missing = checkMissingParameters(function, step.Parameters)
	}

	// Check if we have all parameters now
	if len(step.Missing) == 0 {
		step.Complete = true
		session.CurrentStep = nil

		// Execute the function
		if function, exists := bankingFunctions[step.Intent]; exists {
			result := function.Handler(step.Parameters)
			toolResultChan <- result

			// Stream the result message
			for _, char := range result.Message {
				chunkChan <- string(char)
				time.Sleep(10 * time.Millisecond)
			}
		}
	} else {
		// Ask for next missing parameter
		question := askForMissingParameters(step.Missing, step.Intent)
		for _, char := range question {
			chunkChan <- string(char)
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// ===== Handle Enhanced Non-Streaming Response =====
func handleEnhancedNonStreamingResponse(w http.ResponseWriter, message string, session *UserSession, conversation *ConversationHistory, intent Intent) {
	log.Printf("Handling enhanced non-streaming response for session: %s", session.ID)

	w.Header().Set("Content-Type", "application/json")

	// Process with banking context
	var response string
	var usedTools []string

	// Check for multi-step conversation
	if session.CurrentStep != nil && !session.CurrentStep.Complete {
		response = handleMultiStepConversation(message, intent, conversation, session)
	} else if intent.Confidence > 0.7 && intent.Name != "general" {
		// Handle banking intent
		if function, exists := bankingFunctions[intent.Name]; exists {
			params := extractParametersFromMessage(message, intent, conversation)
			missingParams := checkMissingParameters(function, params)

			if len(missingParams) == 0 {
				result := function.Handler(params)
				usedTools = append(usedTools, function.Name)

				// Get enhanced response from Llama3
				prompt := fmt.Sprintf("%s\n\nUser request: %s\n\nFunction result: %s\n\nProvide a natural response:",
					bankingSystemPrompt, message, result.Message)

				llamaResponse, err := queryLlama3NonStreaming(prompt)
				if err != nil {
					response = result.Message
				} else {
					response = llamaResponse
				}
			} else {
				// Start multi-step
				session.CurrentStep = &ConversationStep{
					StepID:     fmt.Sprintf("step_%d", time.Now().UnixNano()),
					Intent:     intent.Name,
					Parameters: params,
					Missing:    missingParams,
					Complete:   false,
				}
				response = askForMissingParameters(missingParams, intent.Name)
			}
		}
	} else {
		// General query with context
		context := conversation.GetContext()
		prompt := fmt.Sprintf("%s\n\nContext:\n%s\n\nUser: %s\n\nAssistant:",
			bankingSystemPrompt, context, message)

		llamaResponse, err := queryLlama3NonStreaming(prompt)
		if err != nil {
			response = "I'm sorry, I couldn't process your request. Please try again."
		} else {
			response = llamaResponse
		}
	}

	// Add to conversation
	conversation.AddMessage("assistant", response, intent.Name, usedTools, nil)

	json.NewEncoder(w).Encode(ChatResponse{
		Response:  response,
		SessionID: session.ID,
		Intent:    intent.Name,
		Tools:     usedTools,
	})
}

// ===== Handle Multi-Step Conversation (Non-Streaming) =====
func handleMultiStepConversation(message string, intent Intent, conversation *ConversationHistory, session *UserSession) string {
	step := session.CurrentStep

	// Extract information from response
	for _, missing := range step.Missing {
		if value, exists := intent.Entities[missing]; exists {
			step.Parameters[missing] = value
		}
	}

	// Update missing parameters
	if function, exists := bankingFunctions[step.Intent]; exists {
		step.Missing = checkMissingParameters(function, step.Parameters)
	}

	// Check if complete
	if len(step.Missing) == 0 {
		step.Complete = true
		session.CurrentStep = nil

		// Execute function
		if function, exists := bankingFunctions[step.Intent]; exists {
			result := function.Handler(step.Parameters)
			return result.Message
		}
	}

	// Ask for next parameter
	return askForMissingParameters(step.Missing, step.Intent)
}

// ===== Conversation History Handler =====
func conversationHistoryHandler(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract session ID from path
	pathParts := strings.Split(r.URL.Path, "/")
	if len(pathParts) < 4 {
		http.Error(w, "Invalid session ID", http.StatusBadRequest)
		return
	}

	sessionID := pathParts[3]

	// Validate token
	token := r.Header.Get("Authorization")
	if strings.HasPrefix(token, "Bearer ") {
		token = strings.TrimPrefix(token, "Bearer ")
	}

	_, valid := validateToken(token)
	if !valid {
		http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
		return
	}

	conversationMu.RLock()
	conversation, exists := conversationHistory[sessionID]
	conversationMu.RUnlock()

	if !exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"session_id": sessionID,
			"messages":   []ConversationMessage{},
			"context":    map[string]interface{}{},
		})
		return
	}

	conversation.mu.RLock()
	response := map[string]interface{}{
		"session_id": conversation.SessionID,
		"messages":   conversation.Messages,
		"context":    conversation.Context,
	}
	conversation.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ===== Keep your existing functions =====
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// Keep your existing queryLlama3 function
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

// Keep your existing queryLlama3NonStreaming function
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

// Keep your existing streamHandler, pollHandler, and other functions
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

// Keep your existing pollHandler
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

// Keep your existing queryLlama3Streaming
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

// Enhanced cleanup routine with conversation cleanup
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

		// Clean up old conversations
		conversationMu.Lock()
		for sessionID, conv := range conversationHistory {
			if len(conv.Messages) > 0 {
				lastMessage := conv.Messages[len(conv.Messages)-1]
				if now.Sub(lastMessage.Timestamp) > 24*time.Hour {
					delete(conversationHistory, sessionID)
					log.Printf("Cleaned up old conversation %s", sessionID)
				}
			}
		}
		conversationMu.Unlock()

		// Clear old weather cache
		weatherCacheMu.Lock()
		for location, data := range weatherCache {
			if now.Sub(data.LastUpdated) > WeatherCacheExpiry {
				delete(weatherCache, location)
			}
		}
		weatherCacheMu.Unlock()
	}
}
