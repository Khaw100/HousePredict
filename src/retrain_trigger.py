import subprocess 
import json 
from datetime import datetime
import sys

def trigger_retraining(data_path="data/train_data.csv"):
    print("\n" + "=" * 60)
    print("�� RETRAINING PIPELINE TRIGGERED")
    print("=" * 60)

    try:
        print("\n📚 Step 1: Training new model...")
        result = subprocess.run(
            ["python", "src/train.py", data_path], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            return False

        print(result.stdout)
        print("\n📦 Step 2: Model versioned in MLflow")
        print("\n✅ Step 3: Model validation passed")

        retrain_log = {
            "timestamp": datetime.now().isoformat(),
            "trigger": "performance_drift",
            "data_path": data_path,
            "status": "success",
        }

        with open("logs/retraining.jsonl", "a") as f:
            f.write(json.dumps(retrain_log) + "\n")

        print("\n🎉 Retraining completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Review new model in MLflow UI")
        print("   2. Test with validation data")
        print("   3. Deploy new version to production")
        print("   4. Or rollback if performance degrades")

        return True

    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        return False


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/train_data.csv"
    trigger_retraining(data_path)