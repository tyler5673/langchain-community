from typing import List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.you import YouSearchAPIWrapper


class YouSearchInput(BaseModel):
    """Input schema for You.com search."""

    query: str = Field(description="Search query")


YouInput = YouSearchInput


class YouSearchTool(BaseTool):
    """Tool that searches the web using the You.com Search API.

    Setup:
        Set the ``YDC_API_KEY`` environment variable.

        .. code-block:: bash

            export YDC_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools.you import YouSearchTool
            from langchain_community.utilities.you import YouSearchAPIWrapper

            tool = YouSearchTool(
                api_wrapper=YouSearchAPIWrapper(
                    count=5,
                    # livecrawl="web",  # fetch full page content
                    # freshness="week",  # restrict by recency
                )
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'query': 'latest AI research'})

        .. code-block:: python

            [
                Document(
                    page_content="Researchers have developed ...",
                    metadata={
                        "url": "https://example.com/ai-research",
                        "title": "Latest AI Research Breakthroughs",
                        "description": "A summary of recent advances ...",
                        "thumbnail_url": "https://example.com/thumb.jpg",
                        "favicon_url": "https://example.com/favicon.ico",
                        "page_age": "2025-01-15T00:00:00",
                    },
                ),
                ...
            ]

    Invoke with tool call:

        .. code-block:: python

            tool.invoke({"args": {"query": "latest AI research"}, "type": "tool_call", "id": "1", "name": "you_search"})

        .. code-block:: python

            ToolMessage(
                content="[Document(metadata={'url': 'https://example.com/ai-research', 'title': 'Latest AI Research Breakthroughs', ...}, page_content='Researchers have developed ...'), ...]",
                tool_call_id="1",
                name="you_search",
            )

    """  # noqa: E501

    name: str = "you_search"
    description: str = (
        "Search the web using You.com's Search API. Returns factual, "
        "up-to-date web results with snippets and metadata."
    )
    args_schema: Type[BaseModel] = YouSearchInput
    api_wrapper: YouSearchAPIWrapper = Field(default_factory=YouSearchAPIWrapper)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Use the You.com search tool."""
        return self.api_wrapper.results(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Use the You.com search tool asynchronously."""
        return await self.api_wrapper.results_async(query)


class YouContentsInput(BaseModel):
    """Input schema for You.com contents extraction."""

    urls: List[str] = Field(description="URLs to fetch content from")


class YouContentsTool(BaseTool):
    """Tool that fetches clean page content via the You.com Contents API.

    Setup:
        Set the ``YDC_API_KEY`` environment variable.

        .. code-block:: bash

            export YDC_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools.you import YouContentsTool
            from langchain_community.utilities.you import YouSearchAPIWrapper

            tool = YouContentsTool(
                api_wrapper=YouSearchAPIWrapper()
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'urls': ['https://example.com']})

        .. code-block:: python

            [
                Document(
                    page_content="# Example Domain\n\nThis domain is for use in illustrative examples ...",
                    metadata={
                        "url": "https://example.com",
                        "title": "Example Domain",
                        "site_name": "Example",
                        "favicon_url": "https://example.com/favicon.ico",
                    },
                )
            ]

    Invoke with tool call:

        .. code-block:: python

            tool.invoke({"args": {"urls": ["https://example.com"]}, "type": "tool_call", "id": "1", "name": "you_contents"})

        .. code-block:: python

            ToolMessage(
                content="[Document(metadata={'url': 'https://example.com', 'title': 'Example Domain', ...}, page_content='# Example Domain\n\nThis domain is for use in ...')]",
                tool_call_id="1",
                name="you_contents",
            )

    """  # noqa: E501

    name: str = "you_contents"
    description: str = (
        "Fetch clean HTML or Markdown content from web pages using "
        "You.com's Contents API. Useful for extracting readable page content."
    )
    args_schema: Type[BaseModel] = YouContentsInput
    api_wrapper: YouSearchAPIWrapper = Field(default_factory=YouSearchAPIWrapper)

    def _run(
        self,
        urls: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Fetch page content using the You.com Contents API."""
        return self.api_wrapper.contents(urls)

    async def _arun(
        self,
        urls: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Fetch page content asynchronously using the You.com Contents API."""
        return await self.api_wrapper.contents_async(urls)
