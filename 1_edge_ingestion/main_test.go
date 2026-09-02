package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestAddGaussianNoise(t *testing.T) {
	initialVal := 100.0
	iterations := 1000
	sum := 0.0

	for i := 0; i < iterations; i++ {
		noisy := addGaussianNoise(initialVal)
		sum += noisy
		// Ensure noise stays within reasonable physical bounds (e.g. ±10% max for 5*stdDev)
		if math.Abs(noisy-initialVal) > 10.0 {
			t.Errorf("Noisy value %f diverged too far from initial %f", noisy, initialVal)
		}
	}

	mean := sum / float64(iterations)
	// Check mean is close to initial value within 0.5% margin over 1000 samples
	if math.Abs(mean-initialVal) > 1.0 {
		t.Errorf("Mean of noisy distribution %f drifted significantly from initial %f", mean, initialVal)
	}

	// Zero input edge case
	if zeroNoisy := addGaussianNoise(0.0); zeroNoisy != 0.0 {
		t.Errorf("Expected 0.0 for zero input, got %f", zeroNoisy)
	}
}

func TestIngestHandler_ValidPayload(t *testing.T) {
	reqBody := []byte(`{"biomass": 2.5, "glucose": 15.0, "lactate": 0.5}`)
	req, err := http.NewRequest("POST", "/ingest", bytes.NewBuffer(reqBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(ingestHandler)

	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusAccepted {
		t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusAccepted)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse response JSON: %v", err)
	}

	if response["status"] != "ACCEPTED" {
		t.Errorf("Expected response status ACCEPTED, got %v", response["status"])
	}

	dataMap, ok := response["data"].(map[string]interface{})
	if !ok {
		t.Fatalf("Response data field is missing or not a map")
	}

	if dataMap["tank_id"] != "omnicell-sim-01" {
		t.Errorf("Expected tank_id omnicell-sim-01, got %v", dataMap["tank_id"])
	}

	if dataMap["status"] != "RUNNING" {
		t.Errorf("Expected payload status RUNNING, got %v", dataMap["status"])
	}

	// Verify timestamp is valid ISO-8601
	tsStr, ok := dataMap["timestamp"].(string)
	if !ok {
		t.Fatalf("Timestamp field missing or not a string")
	}
	if _, err := time.Parse(time.RFC3339, tsStr); err != nil {
		t.Errorf("Timestamp %s is not valid RFC3339 / ISO-8601: %v", tsStr, err)
	}
}

func TestIngestHandler_InvalidMethod(t *testing.T) {
	req, err := http.NewRequest("GET", "/ingest", nil)
	if err != nil {
		t.Fatalf("Failed to create GET request: %v", err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(ingestHandler)
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405 MethodNotAllowed, got %v", rr.Code)
	}
}

func TestIngestHandler_InvalidJSON(t *testing.T) {
	reqBody := []byte(`{"biomass": "invalid_number"}`)
	req, err := http.NewRequest("POST", "/ingest", bytes.NewBuffer(reqBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(ingestHandler)
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 BadRequest, got %v", rr.Code)
	}
}
