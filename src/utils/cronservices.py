from croniter import croniter
from datetime import datetime
import logging
import os
import threading
import time

from config import CRON_PATH


class CronService:

    def __init__(self, agent):
        self.crons = self._parse_all_crons()
        self.agent = agent

    def _parse_all_crons(self):
        parsed_crons = []
        for filename in os.listdir(CRON_PATH):
            # Get full path
            file_path = os.path.join(CRON_PATH, filename)

            # Ensure it's a file
            if os.path.isfile(file_path) and filename.endswith(".cron"):
                logging.info(f"Parsing cron file {filename}")
                parsed_cron = self._parse_cron_file(file_path)
                if parsed_cron:
                    parsed_crons.append({"filename": filename, **parsed_cron})

        return parsed_crons

    @staticmethod
    def _parse_cron_file(file_path):
        # Open the file and read lines
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Extract the schedule (first line)
        schedule = lines[0].strip()

        # Extract the prompt message (remaining lines)
        prompt_message = "".join(lines[1:]).strip()

        return {"schedule": schedule, "prompt": prompt_message, "name": file_path}

    def run(self):
        for cron in self.crons:
            thread = threading.Thread(
                target=self._schedule_task,
                args=[cron],
                daemon=True,
            )
            thread.start()

    def _schedule_task(self, config):
        cron = croniter(config["schedule"], datetime.now())

        # Calculate the next run time
        next_run_time = cron.get_next(datetime)

        logging.debug(
            f"Scheduler started for job {config['name']}. First run at: {next_run_time}"
        )

        # Keep checking and running the task
        while True:
            current_time = datetime.now()

            # Check if it's time to run the task
            if current_time >= next_run_time:
                self.agent.invoke(config["prompt"])

                # Calculate the next run time
                next_run_time = cron.get_next(datetime)
                logging.debug(
                    f"Next run for job {config['name']} scheduled at: {next_run_time}"
                )

            # Sleep briefly to avoid busy waiting
            time.sleep(10)
