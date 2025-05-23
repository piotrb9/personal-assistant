from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, Output
from langflow.schema import Data
import requests

class WakeOnLanComponent(Component):
    display_name = "Wake-on-LAN"
    description = "Sends a Wake-on-LAN packet to a given MAC address using Home Assistant."
    icon = "power"
    name = "WakeOnLan"

    inputs = [
        StrInput(
            name="mac",
            display_name="MAC Address",
            info="Target device's MAC address to wake.",
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
        StrInput(
            name="broadcast_address",
            display_name="Broadcast Address (Optional)",
            info="Optional broadcast address, e.g., 192.168.1.255.",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Wake-on-LAN Result",
            method="send_wake_packet",
            info="Response from Home Assistant or error details.",
        )
    ]

    def send_wake_packet(self) -> list[Data]:
        try:
            payload = {"mac": self.mac}
            if self.broadcast_address:
                payload["broadcast_address"] = self.broadcast_address

            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                f"{self.ha_url}/api/services/wake_on_lan/send_magic_packet",
                headers=headers,
                json=payload
            )

            result = {
                "status_code": response.status_code,
                "response": response.json() if response.content else "No content"
            }
            self.status = "Wake-on-LAN packet sent successfully."
            return [Data(data=result)]

        except Exception as e:
            self.status = f"Error sending WOL packet: {e}"
            self.log(f"Error: {e}")
            return [Data(data={"error": str(e)})]
