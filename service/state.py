from threading import Lock
import datetime

state_lock = Lock()

STATE = {
    "ready": False,
    "pipeline_status": "idle",
    "data_version": "v0",
    "model_version": "v0.0",
    "trained_on": "v0",
    "last_action": "none",
    
    # Monitoring
    "drift_detected": False,
    "drift_score": 0.0,
    
    # Data Quality
    "data_quality": {
        "valid": True,
        "checks_total": 0,
        "checks_passed": 0,
        "checks_failed": 0,
        "failed_checks": []
    },
    
    # MLflow history
    "experiments": []
}