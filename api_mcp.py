import os
from mcp.server.fastmcp import FastMCP

'''
API MCP Server will have tooling for generating API calls to NewsAPI

The Shape of the API request body is as follows:
{
  "action": "getArticles",
  "keyword": "Tesla Inc",
  "sourceLocationUri": [
    "http://en.wikipedia.org/wiki/United_States",
    "http://en.wikipedia.org/wiki/Canada",
    "http://en.wikipedia.org/wiki/United_Kingdom"
  ],
  "ignoreSourceGroupUri": "paywall/paywalled_sources",
  "articlesPage": 1,
  "articlesCount": 100,
  "articlesSortBy": "date",
  "articlesSortByAsc": false,
  "dataType": [
    "news",
    "pr"
  ],
  "forceMaxDataTimeWindow": 31,
  "resultType": "articles",
  "apiKey": "YOUR_API_KEY"
}
'''

