#!/bin/bash

# Quick smoke test for monitoring system

echo "🧪 Quick Monitoring Test"
echo "========================="

# 1. Check services
echo "
1️⃣ Checking services..."
docker-compose ps

# 2. Generate test predictions
echo "
2️⃣ Generating test predictions (drift scenario)..."
docker-compose exec backend python /src/generate_test_predictions.py drift_variance 30

# 3. Run monitoring
echo "
3️⃣ Running monitoring..."
docker-compose exec backend python /src/monitor.py once

# 4. Check logs
echo "
4️⃣ Checking logs..."
echo "Predictions logged:"
docker-compose exec backend wc -l /app/logs/predictions.jsonl

echo "
Alerts:"
docker-compose exec backend cat /app/logs/alerts.jsonl 2>/dev/null || echo "No alerts"

echo "
✅ Test completed!"
echo "
Next steps:"
echo "  - View MLflow: http://localhost:5000"
echo "  - API health: http://localhost:5001/health"
echo "  - Full test: ./scripts/test_monitoring_workflow.sh"