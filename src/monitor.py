import json 
import pandas as pd
import numpy as np
import sys

from datetime import datetime

class ModelMonitor:
    def __init__(
            self, 
            threshold_variance=100000,
            threshold_samples=20,
            threshold_mae_increase=1.3
    ):
        self.threshold_variance = threshold_variance
        self.threshold_samples = threshold_samples
        self.threshold_mae_increase = threshold_mae_increase
        self.alert_log = []

    def check_performance_drift(self, prediction_file="logs/predictions.jsonl"):

        try:
            # Read recent predictions
            predictions= []
            with open(prediction_file, "r") as f:
                for line in f:
                    predictions.append(json.loads(line))

            if len(predictions) < self.threshold_samples:
                print(
                    f"⏳ Not enough samples yet: {len(predictions)}/{self.threshold_samples}"
                )
                return False, None
            
            # Get recent predictions (last N samples)
            recent = predictions[-self.threshold_samples :]

            # Calculate stats
            pred_values = [p["prediction"] for p in recent]

            mean_pred = np.mean(pred_values)
            std_pred = np.std(pred_values)

            # Load baseline model metrics
            try:
                with open("logs/latest_metrics.json", "r") as f:
                    model_metrics = json.load(f)
                baseline_mae = model_metrics.get("mae", 0)
            except:
                # Set as default
                baseline_mae = 50000 

            print(f"\n📊 Monitoring Report:")
            print(f"   Total predictions: {len(predictions)}")
            print(f"   Analyzing last: {len(recent)} predictions")
            print(f"   Mean prediction: ${mean_pred:,.2f}")
            print(f"   Std prediction: ${std_pred:,.2f}")
            print(f"   Baseline MAE: ${baseline_mae:,.2f}")
            print(f"   Variance threshold: ${self.threshold_variance:,.2f}")

            # Check multiple drift indicators
            drift_detected = False
            drift_reasons = []

            # 1. Check Variance (High Uncertainty)
            if std_pred > self.threshold_variance:
                drift_reasons.append(f"High variance ${std_pred:.2f} > ${self.threshold_variance:,.2f}"
                )
        
            # 2. Check if predictions are too high/low (distribution shift)
            if mean_pred > 500000 or mean_pred < 100000:
                drift_reasons.append(f"Prediction distribution shift: ${mean_pred:,.2f}"
                )
                drift_detected = True

            # 3. Check prediction range
            pred_range = max(pred_values) - min(pred_values)
            if pred_range > 400000:
                drift_reasons.append(f"Large prediction range: ${pred_range:,.2f}")
                drift_detected = True

            # 4. Compare recent predictions with earlier ones (concept drift)
            if len(predictions) >= self.threshold_samples * 2:
                earlier = predictions[-(self.threshold_samples * 2) : -self.threshold_samples ]

                earlier_mean = np.mean([p["prediction"] for p in earlier])
                mean_shift = abs(mean_pred - earlier_mean) / earlier_mean

                if mean_shift > 0.2:  # 20% shift
                    drift_reasons.append(
                        f"Concept drift: {mean_shift*100:.1f}% shift in predictions"
                    )
                    drift_detected = True

            if drift_detected:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "reasons": drift_reasons,
                    "metrics": {
                        "mean_prediction": float(mean_pred),
                        "std_prediction": float(std_pred),
                        "prediction_count": len(predictions),
                        "baseline_mae": float(baseline_mae),
                    },
                }
                self.alert_log.append(alert)

                print(f"\n⚠️  DRIFT DETECTED - Retraining recommended!")
                print(f"   Reasons:")
                for reason in drift_reasons:
                    print(f"   - {reason}")

                return True, alert

            print(f"\n✅ Model performance is stable")
            return False, None
        
        except FileNotFoundError:
            print(
                "There is no predictions log found"
            )
        except Exception as e:
            print(f"Error monitoring: {e}")
            import traceback

            traceback.print_exc()
            return False, None

    def save_alert(self, alert):
        if alert:
            with open("logs/alerts.jsonl", "a") as f:
                f.write(json.dumps(alert) + "\n")
            print(f"\n💾 Alert saved to logs/alerts.jsonl")

def main():
    # Get mode from arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"

    monitor = ModelMonitor(
        threshold_variance=100000, threshold_samples=20  # Lebih rendah, lebih sensitif
    )

    print("🔍 Starting Model Monitor...")
    print("=" * 60)

    if mode == "once":
        # Run once and exit (untuk demo)
        needs_retraining, alert = monitor.check_performance_drift()

        if needs_retraining:
            monitor.save_alert(alert)
            print("\n" + "=" * 60)
            print("🚨 ACTION REQUIRED: Run retraining")
            print("   Command: python src/retrain_trigger.py data/drift_data.csv")
            print("=" * 60)
            sys.exit(1)  # Exit with code 1 untuk indicate action needed
        else:
            print("\n" + "=" * 60)
            print("✅ No action needed")
            print("=" * 60)
            sys.exit(0)

    else:
        # Continuous monitoring (untuk production)
        import time

        while True:
            needs_retraining, alert = monitor.check_performance_drift()

            if needs_retraining:
                monitor.save_alert(alert)
                print("\n🔄 Triggering retraining pipeline...")
                break

            print(f"\n⏰ Next check in 30 seconds...")
            print("=" * 60)
            time.sleep(30)


if __name__ == "__main__":
    main()
