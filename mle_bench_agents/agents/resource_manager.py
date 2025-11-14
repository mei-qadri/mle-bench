"""
Resource Manager Agent (A7) - Continuous Resource Monitoring

Monitors CPU, GPU, memory, disk, and time usage throughout execution.
"""

import time
import psutil
from typing import Any, Dict, Optional
from threading import Thread, Event

from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message, MessageType, create_broadcast_message


class ResourceManagerAgent(Agent):
    """
    Resource Manager Agent responsible for:
    - Monitoring CPU, GPU, memory, disk usage
    - Time budget tracking
    - Alert generation when thresholds exceeded
    - Optimization recommendations
    - Critical action taking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="resource_manager", **kwargs)

        # Monitoring configuration
        self.check_interval = self.config.get("check_interval", 30)  # seconds

        # Resource thresholds
        self.thresholds = self.config.get("thresholds", {
            "cpu": {"warning": 90, "critical": 95},
            "memory": {"warning": 80, "critical": 85},
            "gpu": {"warning": 90, "critical": 95},
            "disk": {"warning": 85, "critical": 90},
            "time_remaining": {"warning": 1800, "critical": 600}
        })

        # Monitoring thread
        self._monitoring_thread: Optional[Thread] = None
        self._stop_event = Event()

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        payload = message.payload
        action = payload.get("action")

        if action == "start_monitoring":
            self.start_monitoring()
            return message.create_response(
                sender_id=self.agent_id,
                payload={"status": "monitoring_started"}
            )

        elif action == "stop_monitoring":
            self.stop_monitoring()
            return message.create_response(
                sender_id=self.agent_id,
                payload={"status": "monitoring_stopped"}
            )

        elif action == "get_metrics":
            metrics = self.collect_metrics()
            return message.create_response(
                sender_id=self.agent_id,
                payload={"metrics": metrics}
            )

        return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource monitoring."""
        self.start_monitoring()

        # Monitoring runs in background thread
        return {"status": "monitoring_active"}

    def start_monitoring(self) -> None:
        """Start monitoring in background thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.log_warning("Monitoring already active")
            return

        self.log_info("Starting resource monitoring")
        self._stop_event.clear()

        self._monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.log_info("Stopping resource monitoring")
        self._stop_event.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            # Collect metrics
            metrics = self.collect_metrics()

            # Update state
            self.update_global_state({"resource_usage": metrics})

            # Check thresholds
            alerts = self.check_thresholds(metrics)

            # Send alerts
            for alert in alerts:
                self.send_alert(alert)

                # Take critical actions
                if alert.get("priority", 0) >= 9:
                    self.take_critical_action(alert)

            # Sleep until next check
            self._stop_event.wait(self.check_interval)

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect resource metrics."""
        global_state = self.get_global_state()

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_used_gb = mem.used / (1024**3)

        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024**3)

        # GPU (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
            else:
                gpu_percent = 0
                gpu_memory_percent = 0
        except:
            gpu_percent = 0
            gpu_memory_percent = 0

        # Time
        time_elapsed = global_state.time_elapsed
        time_remaining = global_state.time_remaining
        time_used_percent = global_state.time_used_percent

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "gpu_percent": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "disk_percent": disk_percent,
            "disk_free_gb": disk_free_gb,
            "time_elapsed": time_elapsed,
            "time_remaining": time_remaining,
            "time_used_percent": time_used_percent
        }

    def check_thresholds(self, metrics: Dict[str, Any]) -> list:
        """Check if metrics exceed thresholds."""
        alerts = []

        # CPU
        if metrics["cpu_percent"] > self.thresholds["cpu"]["critical"]:
            alerts.append({
                "type": "high_cpu",
                "priority": 9,
                "message": f"CPU usage critical: {metrics['cpu_percent']:.1f}%",
                "recommendation": "reduce_parallel_processes"
            })

        # Memory
        if metrics["memory_percent"] > self.thresholds["memory"]["critical"]:
            alerts.append({
                "type": "high_memory",
                "priority": 10,
                "message": f"Memory usage critical: {metrics['memory_percent']:.1f}%",
                "recommendation": "clear_caches_and_reduce_batch_size",
                "action_required": True
            })

        # GPU Memory
        if metrics["gpu_memory_percent"] > self.thresholds["gpu"]["critical"]:
            alerts.append({
                "type": "high_gpu_memory",
                "priority": 9,
                "message": f"GPU memory critical: {metrics['gpu_memory_percent']:.1f}%",
                "recommendation": "clear_cuda_cache",
                "action_required": True
            })

        # Disk
        if metrics["disk_free_gb"] < 5:
            alerts.append({
                "type": "low_disk_space",
                "priority": 8,
                "message": f"Low disk space: {metrics['disk_free_gb']:.1f}GB free",
                "recommendation": "cleanup_temp_files"
            })

        # Time
        if metrics["time_remaining"] < self.thresholds["time_remaining"]["critical"]:
            alerts.append({
                "type": "time_critical",
                "priority": 10,
                "message": f"Time critical: {metrics['time_remaining']}s remaining",
                "recommendation": "finalize_submission_immediately",
                "action_required": True
            })

        return alerts

    def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert message."""
        alert_message = create_broadcast_message(
            sender_id=self.agent_id,
            message_type=MessageType.ALERT,
            payload={
                "alert_type": alert["type"],
                "priority": alert["priority"],
                "message": alert["message"],
                "recommendation": alert.get("recommendation"),
                "action_required": alert.get("action_required", False)
            },
            priority=alert["priority"]
        )

        self.send_message(alert_message)

    def take_critical_action(self, alert: Dict[str, Any]) -> None:
        """Take immediate action for critical alerts."""
        alert_type = alert["type"]

        if alert_type == "high_memory":
            # Clear caches
            import gc
            gc.collect()

            try:
                import torch
                torch.cuda.empty_cache()
                self.log_info("Cleared CUDA cache")
            except:
                pass

        elif alert_type == "time_critical":
            self.log_warning("Time critical - sending emergency alert to orchestrator")
            # Force transition to submission phase
