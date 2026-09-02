package main

import (
	"context"
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/segmentio/kafka-go"
)

// IngestRequest represents the clean incoming telemetry payload from simulation or hardware.
type IngestRequest struct {
	Biomass float64 `json:"biomass"`
	Glucose float64 `json:"glucose"`
	Lactate float64 `json:"lactate"`
}

// EdgePayload represents the processed telemetry message enriched with Raman jitter noise.
type EdgePayload struct {
	Timestamp   string  `json:"timestamp"`
	TankID      string  `json:"tank_id"`
	BiomassGL   float64 `json:"biomass_gL"`
	GlucoseGL   float64 `json:"glucose_gL"`
	LactateMmolL float64 `json:"lactate_mmolL"`
	Status      string  `json:"status"`
}

// Global Kafka Writer connection pool manager.
var kafkaWriter *kafka.Writer

func init() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	broker := os.Getenv("KAFKA_BROKER")
	if broker == "" {
		broker = "localhost:9092"
	}

	topic := os.Getenv("KAFKA_TOPIC")
	if topic == "" {
		topic = "omnicell-telemetry"
	}

	// Initialize Kafka Writer globally to reuse connection pool and avoid connection churn.
	kafkaWriter = &kafka.Writer{
		Addr:                   kafka.TCP(broker),
		Topic:                  topic,
		Balancer:               &kafka.LeastBytes{},
		RequiredAcks:           kafka.RequireOne,
		AllowAutoTopicCreation: true,
		Async:                  false, // Explicit goroutine dispatch used in HTTP handler for async control
	}

	log.Printf("[Init] Kafka writer initialized for broker=%s topic=%s", broker, topic)
}

// addGaussianNoise adds synthetic Gaussian noise distribution (±2% standard deviation)
// to simulate physical Raman spectrometer hardware jitter.
func addGaussianNoise(val float64) float64 {
	if val == 0 {
		return 0
	}
	stdDev := math.Abs(val) * 0.02
	noisy := val + rand.NormFloat64()*stdDev
	if noisy < 0 {
		return 0 // Enforce non-negative physical boundary condition
	}
	return math.Round(noisy*10000) / 10000 // Round to 4 decimal places for precision stability
}

// publishTelemetryAsync publishes the payload byte slice to Kafka in a non-blocking goroutine.
func publishTelemetryAsync(payloadBytes []byte) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		err := kafkaWriter.WriteMessages(ctx, kafka.Message{
			Key:   []byte("omnicell-sim-01"),
			Value: payloadBytes,
			Time:  time.Now(),
		})

		if err != nil {
			log.Printf("[Kafka Error] Failed to publish message: %v", err)
		} else {
			log.Printf("[Kafka Success] Telemetry event published (%d bytes)", len(payloadBytes))
		}
	}()
}

// ingestHandler processes POST /ingest requests asynchronously.
func ingestHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error": "Method not allowed. Use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	var req IngestRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{
			"error":   "Invalid JSON payload",
			"details": err.Error(),
		})
		return
	}

	// Apply Raman spectrometer Gaussian jitter
	noisyBiomass := addGaussianNoise(req.Biomass)
	noisyGlucose := addGaussianNoise(req.Glucose)
	noisyLactate := addGaussianNoise(req.Lactate)

	// Map into EdgePayload struct
	payload := EdgePayload{
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		TankID:       "omnicell-sim-01",
		BiomassGL:    noisyBiomass,
		GlucoseGL:    noisyGlucose,
		LactateMmolL: noisyLactate,
		Status:       "RUNNING",
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to serialize payload"})
		return
	}

	// Dispatch Kafka publish asynchronously without blocking HTTP response
	publishTelemetryAsync(payloadBytes)

	// Immediately respond to caller
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "ACCEPTED",
		"message": "Telemetry queued for Kafka ingestion",
		"data":    payload,
	})
}

// healthHandler provides health status for monitoring.
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "UP",
		"engine": "OmniCell Edge Ingestion v1.0",
	})
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	http.HandleFunc("/ingest", ingestHandler)
	http.HandleFunc("/health", healthHandler)

	server := &http.Server{
		Addr:         ":" + port,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Channel to listen for interrupt signals for graceful shutdown
	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("[Server] Starting Edge Ingestion Engine on port %s...", port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[Server Error] %v", err)
		}
	}()

	<-stopChan
	log.Println("[Server] Shutting down Edge Ingestion Engine...")

	// Gracefully shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("[Server] HTTP shutdown error: %v", err)
	}

	// Close global Kafka writer pool
	if err := kafkaWriter.Close(); err != nil {
		log.Printf("[Kafka] Error closing writer: %v", err)
	} else {
		log.Println("[Kafka] Connection pool closed cleanly")
	}

	log.Println("[Server] Shutdown complete.")
}
