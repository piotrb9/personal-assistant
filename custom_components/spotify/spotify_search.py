from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, IntInput, DropdownInput, Output
from langflow.schema import Data
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifySearchComponent(Component):
    display_name = "Spotify Search (Client Credentials)"
    description = "Searches Spotify using Client Credentials Flow."
    icon = "music"
    name = "SpotifySearchClientCredentials"

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
            required=True
        ),
        StrInput(
            name="query",
            display_name="Search Query",
            info="The text query to search for.",
            required=True,
            tool_mode=True
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            options=["artist", "album", "track", "playlist", "show", "episode"],
            value="track",
            info="The type of item to search for.",
            required=True,
            tool_mode=True
        ),
        IntInput(
            name="limit",
            display_name="Limit",
            info="The maximum number of results to return.",
            value=10,
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="results",
            display_name="Search Results",
            method="search_spotify",
            info="List of search results as Data objects.",
        )
    ]

    def search_spotify(self) -> list[Data]:
        """Performs a search on Spotify and returns results."""
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            sp = spotipy.Spotify(auth_manager=auth_manager)

            results = sp.search(
                q=self.query,
                type=self.search_type,
                limit=self.limit
            )

            items = results.get(f'{self.search_type}s', {}).get('items', [])
            formatted_results = []
            for item in items:
                data_content = {"spotify_id": item.get("id"), "name": item.get("name")}
                if self.search_type == "track":
                    artists = ", ".join([artist['name'] for artist in item.get("artists", [])])
                    data_content["artists"] = artists
                    data_content["album"] = item.get("album", {}).get("name")
                    # print(f"Found Track: {item.get('name')} by {artists}")
                elif self.search_type == "artist":
                    data_content["genres"] = ", ".join(item.get("genres", []))
                    # print(f"Found Artist: {item.get('name')}")

                formatted_results.append(Data(data=data_content))

            self.status = f"Successfully found {len(formatted_results)} {self.search_type}(s) for query '{self.query}'."
            return formatted_results

        except Exception as e:
            self.status = f"Error during Spotify search: {e}"
            self.log(f"Error: {e}")
            return [Data(data={"error": str(e)})]
