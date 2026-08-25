"""
providers 包。

Provider 负责从具体数据源读取数据。
当前数据源是 data/mock/*.json，用于模拟外部查询结果。
"""

from app.providers.flight_mock_provider import search_flights_from_mock
from app.providers.hotel_mock_provider import search_hotels_from_mock
from app.providers.weather_mock_provider import query_weather_from_mock

__all__ = [
    "query_weather_from_mock",
    "search_flights_from_mock",
    "search_hotels_from_mock",
]