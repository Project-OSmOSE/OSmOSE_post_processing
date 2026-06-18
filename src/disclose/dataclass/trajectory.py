"""`trajectory` module provides `Trajectory` dataclass for interpolating positions."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class Trajectory:
    """Handles trajectory interpolation for a series of timestamps and positions."""

    def __init__(self) -> None:
        """Initialize an empty trajectory."""
        self.timestamps: np.ndarray = np.array([])
        self.latitudes: np.ndarray = np.array([])
        self.longitudes: np.ndarray = np.array([])
        self.lat_interp: interp1d | None = None
        self.lon_interp: interp1d | None = None

    def add_position(self, timestamp: float, latitude: float, longitude: float) -> None:
        """Add a new position to the trajectory.

        Ignores duplicates in timestamp or identical consecutive coordinates.

        Parameters
        ----------
        timestamp : float
            Unix timestamp or any numeric representation of time.
        latitude : float
            Latitude of the position in decimal degrees.
        longitude : float
            Longitude of the position in decimal degrees.

        """
        if self.timestamps.size > 0:
            if timestamp == self.timestamps[-1]:
                return
            if latitude == self.latitudes[-1] and longitude == self.longitudes[-1]:
                return

        self.timestamps = np.append(self.timestamps, timestamp)
        self.latitudes = np.append(self.latitudes, latitude)
        self.longitudes = np.append(self.longitudes, longitude)

        # Update linear interpolators
        if self.timestamps.size >= 2:  # noqa: PLR2004
            self.lat_interp = interp1d(
                self.timestamps,
                self.latitudes,
                kind="linear",
                fill_value="extrapolate",
            )
            self.lon_interp = interp1d(
                self.timestamps,
                self.longitudes,
                kind="linear",
                fill_value="extrapolate",
            )

    def get_position(self, timestamp: float) -> [float, float]:
        """Get interpolated latitude and longitude for a given timestamp.

        Parameters
        ----------
        timestamp : float
            The timestamp at which to interpolate the position.

        Returns
        -------
        [float, float]
            Interpolated latitude and longitude.
            Returns (nan, nan) if insufficient data.

        """
        if self.lat_interp is None or self.lon_interp is None:
            return float("nan"), float("nan")
        lat: float = float(self.lat_interp(timestamp))
        lon: float = float(self.lon_interp(timestamp))
        return lat, lon
