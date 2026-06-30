"""
Search Agent - handles web searches and information retrieval
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class SearchAgent:
    """Agent for search operations"""
    
    def __init__(self, serper_api_key=None):
        self.search_history = []
        self.serper_api_key = serper_api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self.logger = logger
    
    async def web_search(self, query: str, num_results: int = 5, language: str = 'en') -> Dict[str, Any]:
        """Perform web search using Serper API"""
        try:
            # Use Serper API if key is available
            if self.serper_api_key:
                url = "https://google.serper.dev/search"
                
                payload = {
                    "q": query,
                    "num": num_results,
                    "hl": language
                }
                
                headers = {
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json"
                }
                
                response = await self.client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                results = response.json()
                
                # Extract organic results from Serper response
                organic_results = results.get("organic", [])
                formatted_results = []
                
                for result in organic_results[:num_results]:
                    formatted_results.append({
                        'rank': len(formatted_results) + 1,
                        'title': result.get('title', ''),
                        'url': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': result.get('link', '').split('/')[2] if 'link' in result else 'unknown'
                    })
                
                search_result = {
                    'id': len(self.search_history) + 1,
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'num_results': len(formatted_results),
                    'language': language,
                    'results': formatted_results,
                    'status': 'completed'
                }
                self.search_history.append(search_result)
                
                self.logger.info(f"Web search performed with Serper: {query}")
                return {
                    'success': True,
                    'query': query,
                    'results_count': len(formatted_results),
                    'results': formatted_results
                }
            else:
                # Fallback to mock results
                search_result = {
                    'id': len(self.search_history) + 1,
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'num_results': num_results,
                    'language': language,
                    'results': self._generate_mock_results(query, num_results),
                    'status': 'completed'
                }
                self.search_history.append(search_result)
                
                self.logger.info(f"Web search performed (mock): {query}")
                return {
                    'success': True,
                    'query': query,
                    'results_count': len(search_result['results']),
                    'results': search_result['results']
                }
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def search_news(self, topic: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for news"""
        try:
            news_results = {
                'topic': topic,
                'timestamp': datetime.now().isoformat(),
                'articles': self._generate_mock_news(topic, num_results),
                'total_articles': num_results
            }
            
            self.logger.info(f"News search performed: {topic}")
            return {
                'success': True,
                'topic': topic,
                'articles': news_results['articles']
            }
        except Exception as e:
            self.logger.error(f"Error searching news: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_mock_results(self, query: str, num_results: int) -> list:
        """Generate mock search results"""
        results = []
        for i in range(num_results):
            results.append({
                'rank': i + 1,
                'title': f'{query} - Result {i + 1}',
                'url': f'https://example.com/result{i + 1}',
                'snippet': f'This is a search result snippet for "{query}". Position {i + 1}.',
                'source': f'source{i + 1}.com'
            })
        return results
    
    def _generate_mock_news(self, topic: str, num_articles: int) -> list:
        """Generate mock news articles"""
        articles = []
        for i in range(num_articles):
            articles.append({
                'id': i + 1,
                'title': f'{topic} News - Article {i + 1}',
                'source': f'news{i + 1}.com',
                'published_at': datetime.now().isoformat(),
                'summary': f'This is a news summary about {topic}. Article {i + 1}.',
                'url': f'https://example.com/news{i + 1}'
            })
        return articles
    
    async def get_search_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get search history"""
        try:
            recent_searches = self.search_history[-limit:] if self.search_history else []
            return {
                'success': True,
                'search_count': len(recent_searches),
                'searches': recent_searches
            }
        except Exception as e:
            self.logger.error(f"Error getting search history: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute search action"""
        if action == 'web':
            return await self.web_search(**kwargs)
        elif action == 'news':
            return await self.search_news(**kwargs)
        elif action == 'history':
            return await self.get_search_history(**kwargs)
        else:
            return {'success': False, 'error': f'Unknown action: {action}'}
