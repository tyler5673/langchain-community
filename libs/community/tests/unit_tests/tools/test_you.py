import warnings
from unittest.mock import AsyncMock, patch

import responses

from langchain_community.tools.you import YouContentsTool, YouSearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper

from ..utilities.test_you import (
    LIMITED_PARSED_OUTPUT,
    MOCK_CONTENTS_PARSED,
    MOCK_CONTENTS_RESPONSE,
    MOCK_PARSED_OUTPUT,
    MOCK_RESPONSE_RAW,
    NEWS_RESPONSE_PARSED,
    NEWS_RESPONSE_RAW,
    TEST_ENDPOINT,
)


class TestYouSearchTool:
    @responses.activate
    def test_invoke(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=MOCK_RESPONSE_RAW,
            status=200,
        )
        query = "Test query text"
        you_tool = YouSearchTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="test"))
        results = you_tool.invoke(query)
        assert results == MOCK_PARSED_OUTPUT

    @responses.activate
    def test_invoke_max_docs(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=MOCK_RESPONSE_RAW,
            status=200,
        )
        query = "Test query text"
        you_tool = YouSearchTool(
            api_wrapper=YouSearchAPIWrapper(ydc_api_key="test", k=2)
        )
        results = you_tool.invoke(query)
        expected_result = [MOCK_PARSED_OUTPUT[0], MOCK_PARSED_OUTPUT[1]]
        assert results == expected_result

    @responses.activate
    def test_invoke_limit_snippets(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=MOCK_RESPONSE_RAW,
            status=200,
        )
        query = "Test query text"
        you_tool = YouSearchTool(
            api_wrapper=YouSearchAPIWrapper(ydc_api_key="test", n_snippets_per_hit=1)
        )
        results = you_tool.invoke(query)
        assert results == LIMITED_PARSED_OUTPUT

    @responses.activate
    def test_invoke_news(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=NEWS_RESPONSE_RAW,
            status=200,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            you_tool = YouSearchTool(
                api_wrapper=YouSearchAPIWrapper(
                    ydc_api_key="test", endpoint_type="news"
                )
            )
        results = you_tool.invoke("Test news text")
        assert results == NEWS_RESPONSE_PARSED

    async def test_ainvoke(self) -> None:
        you_tool = YouSearchTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="test"))

        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MOCK_RESPONSE_RAW)
        mock_response.raise_for_status = lambda: None

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            results = await you_tool.ainvoke("test query")
            assert results == MOCK_PARSED_OUTPUT


class TestYouContentsTool:
    @responses.activate
    def test_invoke(self) -> None:
        responses.add(
            responses.POST,
            f"{TEST_ENDPOINT}/v1/contents",
            json=MOCK_CONTENTS_RESPONSE,
            status=200,
        )
        you_tool = YouContentsTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="test"))
        results = you_tool.invoke({"urls": ["https://example.com"]})
        assert results == MOCK_CONTENTS_PARSED

    async def test_ainvoke(self) -> None:
        you_tool = YouContentsTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="test"))

        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MOCK_CONTENTS_RESPONSE)
        mock_response.raise_for_status = lambda: None

        with patch("aiohttp.ClientSession.post", return_value=mock_response):
            results = await you_tool.ainvoke({"urls": ["https://example.com"]})
            assert results == MOCK_CONTENTS_PARSED
