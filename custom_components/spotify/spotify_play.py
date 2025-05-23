from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, Output
from langflow.schema import Data
import spotipy
from spotipy.oauth2 import SpotifyOAuth

class SpotifyPlayComponent(Component):
    display_name = "Spotify Play (Authorization Code)"
    description = "Plays a track on Spotify using the Authorization Code Flow."
    icon = "play-circle"
    name = "SpotifyPlayAuthorizationCode"

    inputs = [
        StrInput(
            name="client_id",
            display_name="Spotify Client ID",
            info="Your Spotify API Client ID.",
            required=True,
        ),
        SecretStrInput(
            name="client_secret",
            display_name="Spotify Client Secret",
            info="Your Spotify API Client Secret.",
            required=True,
        ),
        StrInput(
            name="redirect_uri",
            display_name="Redirect URI",
            info="Redirect URI set in your Spotify Developer Dashboard.",
            required=True,
        ),
        SecretStrInput(
            name="refresh_token",
            display_name="User Refresh Token",
            info="A previously obtained refresh token for the user (with playback control scopes).",
            required=True,
        ),
        StrInput(
            name="track_uri",
            display_name="Spotify Track URI or ID",
            info="The Spotify URI (e.g., 'spotify:track:ID') or ID of the track to play.",
            required=True,
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Playback Result",
            method="play_track",
            info="Playback result or error as a Data object.",
        )
    ]

    def play_track(self) -> list[Data]:
        try:
            oauth = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope="user-modify-playback-state",
            )
            token_info = oauth.refresh_access_token(self.refresh_token)
            sp = spotipy.Spotify(auth=token_info["access_token"])

            track_uri = self.track_uri
            if not track_uri.startswith("spotify:track:"):
                track_uri = f"spotify:track:{track_uri}"

            sp.start_playback(uris=[track_uri])

            self.status = f"Successfully started playback for {track_uri}."
            return [Data(data={"status": "success", "track_uri": track_uri})]

        except Exception as e:
            self.status = f"Error during playback: {e}"
            self.log(f"Error: {e}")
            return [Data(data={"error": str(e)})]
