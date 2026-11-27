"""Tests for the code search tool (client, server, transformers)."""

import json

import httpx
import pytest
import respx
from unittest.mock import AsyncMock, patch

from src.client import BitbucketClient
from src.utils.transformers import slim_code_search, slim_code_search_file

SEARCH_URL = "https://api.bitbucket.org/2.0/workspaces/workspace/codesearch"

# ========== Fixtures ==========

SAMPLE_FILE = {
    "path": "foundation/job.yaml",
    "name": "job.yaml",
    "language": "yaml",
    "url": "https://bitbucket.org/bscideas/cid-cmpt-configdeployer/src/768f186eb785aa/foundation/job.yaml",
    "repositoryId": "repo-1",
    "version": "768f186eb785aabbccddeeff00112233",
    "matched_chunks": [
        {
            "source_code": "      files:\n      - ->id: //dbos-job->job\n",
            "line_range": {"start": 22, "end": 23},
            "matched_line_numbers": [23],
            "matched_ranges": [[23, 15, 23]],
        }
    ],
}

SAMPLE_RESPONSE = {
    "id": "req-1",
    "pagination": {"has_more": True, "next_cursor": "cur-1"},
    "repositories": [
        {"id": "repo-1", "ns": "bscideas", "name": "cid-cmpt-configdeployer"}
    ],
    "matched_files": [SAMPLE_FILE],
    "metadata": {"total_count": 7},
}


# ========== Transformers ==========

def test_slim_code_search_file_uses_repository_map():
    r = slim_code_search_file(
        {**SAMPLE_FILE, "url": None}, {"repo-1": "bscideas/cid-cmpt-configdeployer"}
    )
    assert r == {
        "repository": "bscideas/cid-cmpt-configdeployer",
        "path": "foundation/job.yaml",
        "commit": "768f186eb785",
        "chunks": [
            {
                "lines": "22-23",
                "matched_lines": [23],
                "code": "      files:\n      - ->id: //dbos-job->job\n",
            }
        ],
    }


def test_slim_code_search_file_falls_back_to_url():
    r = slim_code_search_file(SAMPLE_FILE)
    assert r["repository"] == "bscideas/cid-cmpt-configdeployer"


def test_slim_code_search_file_tolerates_missing_fields():
    r = slim_code_search_file({"path": "README.md"})
    assert r["repository"] is None
    assert r["commit"] is None
    assert r["chunks"] == []


def test_slim_code_search_with_more_pages():
    r = slim_code_search(SAMPLE_RESPONSE)
    assert r["count"] == 1
    assert r["total"] == 7
    assert r["has_more"] is True
    assert r["request_id"] == "req-1"
    assert r["next_cursor"] == "cur-1"
    assert r["values"][0]["repository"] == "bscideas/cid-cmpt-configdeployer"


def test_slim_code_search_last_page_has_no_cursor():
    r = slim_code_search({**SAMPLE_RESPONSE, "pagination": {"has_more": False}})
    assert r["has_more"] is False
    assert "next_cursor" not in r
    assert "request_id" not in r


# ========== Client ==========

def _body(route):
    return json.loads(route.calls.last.request.content)


@pytest.mark.asyncio
async def test_search_code_minimal_body():
    with respx.mock:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            result = await client.search_code("dbos-job")
        assert _body(route) == {
            "query": {"search_term": "dbos-job"},
            "config": {"max_results": 20},
        }
        assert route.calls.last.request.headers["content-type"] == "application/json"
        assert result["id"] == "req-1"


@pytest.mark.asyncio
async def test_search_code_single_filter_wildcard():
    with respx.mock:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            await client.search_code("job", repository="cid-cmpt-*")
        assert _body(route)["query"]["filters"] == {
            "attribute": "repository", "wildcard": "cid-cmpt-*"
        }


@pytest.mark.asyncio
async def test_search_code_combines_filters_with_and():
    with respx.mock:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            await client.search_code(
                "job", repository="cid-cmpt-weblate", path="extras/initialize",
                extension="yaml", language="yaml", project="CID_CMPT",
            )
        assert _body(route)["query"]["filters"] == {
            "and": [
                {"attribute": "repository", "eq": "cid-cmpt-weblate"},
                {"attribute": "path", "eq": "extras/initialize"},
                {"attribute": "extension", "eq": "yaml"},
                {"attribute": "language", "eq": "yaml"},
                {"attribute": "project", "eq": "CID_CMPT"},
            ]
        }


@pytest.mark.asyncio
async def test_search_code_extension_never_wildcard():
    with respx.mock:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            await client.search_code("job", extension="ya?l")
        assert _body(route)["query"]["filters"] == {
            "attribute": "extension", "eq": "ya?l"
        }


@pytest.mark.asyncio
async def test_search_code_next_page():
    with respx.mock:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            await client.search_code(
                "job", max_results=5, request_id="req-1", next_cursor="cur-1"
            )
        body = _body(route)
        assert body["request_id"] == "req-1"
        assert body["pagination"] == {"next_cursor": "cur-1"}
        assert body["config"] == {"max_results": 5}


@pytest.mark.asyncio
async def test_search_code_explicit_workspace():
    with respx.mock:
        route = respx.post(
            "https://api.bitbucket.org/2.0/workspaces/other/codesearch"
        ).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            await client.search_code("foo", workspace="other")
        assert route.called


@pytest.mark.asyncio
async def test_search_code_http_error_propagates():
    with respx.mock:
        respx.post(SEARCH_URL).mock(return_value=httpx.Response(403, json={}))
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.search_code("foo")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"search_term": ""}, "must not be empty"),
        ({"search_term": "   "}, "must not be empty"),
        ({"search_term": "x" * 1025}, "at most 1024"),
        ({"search_term": "x", "max_results": 0}, "between 1 and 100"),
        ({"search_term": "x", "max_results": 101}, "between 1 and 100"),
        ({"search_term": "x", "request_id": "r"}, "together"),
        ({"search_term": "x", "next_cursor": "c"}, "together"),
    ],
)
async def test_search_code_rejects_invalid_input(kwargs, message):
    with respx.mock(assert_all_called=False):
        async with BitbucketClient("e@x.com", "token", "workspace") as client:
            with pytest.raises(ValueError, match=message):
                await client.search_code(**kwargs)


# ========== Server tool ==========

@pytest.mark.asyncio
@patch("src.server.get_client")
async def test_search_code_tool(mock_get_client):
    from src.server import search_code

    mock_client = AsyncMock()
    mock_client.search_code.return_value = SAMPLE_RESPONSE
    mock_get_client.return_value = mock_client

    result = await search_code("dbos-job", repository="cid-cmpt-*")

    mock_client.search_code.assert_awaited_once_with(
        "dbos-job", None, "cid-cmpt-*", None, None, None, None, 20, None, None
    )
    assert result["total"] == 7
    assert result["values"][0]["path"] == "foundation/job.yaml"


@pytest.mark.asyncio
@patch("src.server.get_client")
async def test_search_code_tool_drops_total_on_next_pages(mock_get_client):
    from src.server import search_code

    mock_client = AsyncMock()
    mock_client.search_code.return_value = SAMPLE_RESPONSE
    mock_get_client.return_value = mock_client

    result = await search_code("dbos-job", request_id="req-1", next_cursor="cur-1")

    assert "total" not in result
    assert result["next_cursor"] == "cur-1"
