import threading
import time

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False


class BackupScheduler:
    def __init__(self, schedule_type: str, callback):
        if not SCHEDULE_AVAILABLE:
            return

        if schedule_type == "daily":
            schedule.every().day.at("02:00").do(callback)
        elif schedule_type == "weekly":
            schedule.every().monday.at("02:00").do(callback)
        elif schedule_type == "monthly":
            schedule.every(30).days.at("02:00").do(callback)

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            schedule.run_pending()
            time.sleep(60)
