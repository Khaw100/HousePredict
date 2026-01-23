#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🧪 COMPLETE MONITORING & RETRAINING WORKFLOW TEST         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section header
print_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check if containers are running
check_containers() {
    print_header "🔍 CHECKING DOCKER CONTAINERS"
    
    if ! docker-compose ps | grep -q "Up"; then
        echo -e "${RED}❌ Containers are not running!${NC}"
        echo -e "${YELLOW}Starting containers...${NC}"
        docker-compose up -d
        sleep 5
    else
        echo -e "${GREEN}✅ Containers are running${NC}"
    fi
    
    docker-compose ps
}

# Function to clear old logs
clear_logs() {
    print_header "🗑️  CLEARING OLD LOGS"
    
    docker-compose exec backend rm -f /app/logs/predictions.jsonl 2>/dev/null
    docker-compose exec backend rm -f /app/logs/alerts.jsonl 2>/dev/null
    docker-compose exec backend rm -f /app/logs/retraining.jsonl 2>/dev/null
    
    echo -e "${GREEN}✅ Logs cleared${NC}"
}

# Function to run monitoring test
run_monitoring_test() {
    local test_name=$1
    local scenario=$2
    local expected_drift=$3
    local samples=${4:-50}
    
    print_header "📌 TEST: ${test_name}"
    
    # Clear predictions
    docker-compose exec backend rm -f /app/logs/predictions.jsonl 2>/dev/null
    
    # Generate predictions
    echo -e "${YELLOW}➜ Generating ${samples} predictions (scenario: ${scenario})...${NC}"
    docker-compose exec backend python /src/generate_test_predictions.py "$scenario" "$samples"
    
    echo ""
    echo -e "${YELLOW}➜ Running monitoring...${NC}"
    docker-compose exec backend python /src/monitor.py once
    exit_code=$?
    
    # Check result
    echo ""
    if [ $exit_code -eq 1 ] && [ "$expected_drift" = "yes" ]; then
        echo -e "${GREEN}✅ PASS: Drift detected as expected${NC}"
        return 0
    elif [ $exit_code -eq 0 ] && [ "$expected_drift" = "no" ]; then
        echo -e "${GREEN}✅ PASS: No drift detected as expected${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL: Unexpected result (exit code: $exit_code, expected drift: $expected_drift)${NC}"
        return 1
    fi
}

# Function to test API predictions
test_api_predictions() {
    print_header "🌐 TESTING API PREDICTIONS"
    
    echo -e "${YELLOW}Testing /health endpoint...${NC}"
    curl -s http://localhost:5001/health | python -m json.tool
    
    echo ""
    echo -e "${YELLOW}Testing /predict/single endpoint...${NC}"
    
    # Create test JSON
    cat > /tmp/test_house.json <<EOF
{
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "TotalBsmtSF": 1000,
    "1stFlrSF": 1000,
    "2ndFlrSF": 500,
    "YearBuilt": 2000,
    "YrSold": 2010,
    "BsmtFinSF1": 800,
    "BsmtFinSF2": 0,
    "OpenPorchSF": 50,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "WoodDeckSF": 100
}
EOF
    
    response=$(curl -s -X POST http://localhost:5001/predict/single \
        -H "Content-Type: application/json" \
        -d @/tmp/test_house.json)
    
    echo "$response" | python -m json.tool
    
    if echo "$response" | grep -q "prediction"; then
        echo -e "${GREEN}✅ API prediction successful${NC}"
    else
        echo -e "${RED}❌ API prediction failed${NC}"
    fi
}

# Function to show monitoring stats
show_monitoring_stats() {
    print_header "📊 MONITORING STATISTICS"
    
    echo -e "${YELLOW}Querying /monitoring/stats...${NC}"
    curl -s http://localhost:5001/monitoring/stats | python -m json.tool
}

# Function to show drift alerts
show_drift_alerts() {
    print_header "⚠️  DRIFT ALERTS"
    
    if docker-compose exec backend test -f /app/logs/alerts.jsonl; then
        echo -e "${YELLOW}Alerts found:${NC}"
        docker-compose exec backend cat /app/logs/alerts.jsonl | while read line; do
            echo "$line" | python -m json.tool
            echo ""
        done
    else
        echo -e "${GREEN}✅ No drift alerts (all tests passed normally)${NC}"
    fi
}

# Main test execution
main() {
    # Check prerequisites
    check_containers
    
    # Clear old data
    clear_logs
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  PART 1: MONITORING TESTS                                  ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    # Test 1: Normal predictions (no drift)
    run_monitoring_test \
        "Normal Predictions (Baseline)" \
        "normal" \
        "no" \
        30
    
    sleep 2
    
    # Test 2: High variance drift
    run_monitoring_test \
        "High Variance Drift" \
        "drift_variance" \
        "yes" \
        30
    
    sleep 2
    
    # Test 3: Distribution shift (high)
    run_monitoring_test \
        "Distribution Shift (Too High)" \
        "drift_high" \
        "yes" \
        30
    
    sleep 2
    
    # Test 4: Distribution shift (low)
    run_monitoring_test \
        "Distribution Shift (Too Low)" \
        "drift_low" \
        "yes" \
        30
    
    sleep 2
    
    # Test 5: Concept drift
    run_monitoring_test \
        "Concept Drift Over Time" \
        "concept_drift" \
        "yes" \
        50
    
    # Show drift alerts
    show_drift_alerts
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  PART 2: API INTEGRATION TESTS                             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    # Clear for fresh API test
    docker-compose exec backend rm -f /app/logs/predictions.jsonl 2>/dev/null
    
    # Test API predictions
    test_api_predictions
    
    sleep 2
    
    # Show monitoring stats from API predictions
    show_monitoring_stats
    
    # Final summary
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  📋 TEST SUMMARY                                           ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    echo ""
    echo -e "${GREEN}✅ All tests completed!${NC}"
    echo ""
    
    echo -e "${YELLOW}📁 Generated Files:${NC}"
    echo "   - logs/predictions.jsonl  (prediction logs)"
    echo "   - logs/alerts.jsonl       (drift alerts)"
    echo "   - logs/retraining.jsonl   (retraining history)"
    echo ""
    
    echo -e "${YELLOW}🔗 Useful Commands:${NC}"
    echo "   View predictions:  docker-compose exec backend cat /app/logs/predictions.jsonl"
    echo "   View alerts:       docker-compose exec backend cat /app/logs/alerts.jsonl"
    echo "   Run monitoring:    docker-compose exec backend python /src/monitor.py once"
    echo "   Trigger retrain:   docker-compose exec backend python /src/retrain_trigger.py"
    echo "   MLflow UI:         http://localhost:5000"
    echo "   API Health:        http://localhost:5001/health"
    echo ""
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Run main function
main