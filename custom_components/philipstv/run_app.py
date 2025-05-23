from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, Output
from langflow.schema import Data
import threading
import requests
import time

class RunAppComponent(Component):
    display_name = "Run App on TV"
    description = "Launches an app (e.g., Netflix, YouTube) on a smart TV using Home Assistant."
    icon = "tv"
    name = "RunAppOnTV"

    inputs = [
        StrInput(
            name="app_name",
            display_name="App Name",
            info="App to launch (e.g., 'Netflix', 'YouTube', 'HBO').",
            required=True,
        ),
        StrInput(
            name="entity",
            display_name="Remote Entity ID",
            info="Home Assistant entity ID of the remote (e.g., 'remote.living_room_tv').",
            required=True,
        ),
        SecretStrInput(
            name="token",
            display_name="Home Assistant Token",
            info="Long-lived access token from Home Assistant.",
            required=True,
        ),
        StrInput(
            name="ha_url",
            display_name="Home Assistant URL",
            info="Base URL of your Home Assistant instance (e.g., http://localhost:8123).",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="App Launch Result",
            method="launch_app",
            info="Success or error message.",
        )
    ]

    def launch_app(self) -> list[Data]:
        try:
            app_dict = {
                "HBO": 1,
                "Netflix": 2,
                "YouTube": 3,
            }

            if self.app_name not in app_dict:
                raise ValueError(f"Unsupported app: {self.app_name}")

            def send_command(command):
                payload = {
                    "entity_id": self.entity,
                    "command": command,
                }
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                }
                threading.Thread(
                    target=requests.post,
                    args=(f"{self.ha_url}/api/services/remote/send_command",),
                    kwargs={"headers": headers, "json": payload},
                ).start()

            for _ in range(app_dict[self.app_name]):
                send_command("CursorRight")
                time.sleep(0.3)

            send_command("Confirm")

            self.status = f"{self.app_name} launch command sent successfully."
            return [Data(data={"status": "success", "app": self.app_name})]

        except Exception as e:
            self.status = f"Error: {e}"
            self.log(f"Error: {e}")
            return [Data(data={"error": str(e)})]
