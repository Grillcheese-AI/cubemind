---
auto_test_after_edit: true
external_validation:
  audit_completed_tasks: true
  enabled: false
  model: google/gemini-3.1-flash-lite-preview
  openrouter_api_key: null
  require_approval_on_issues: true
  validate_plans: true
frameworks: []
max_memories: 50000
project:
  business_docs_path: docs/business
  framework: grilly
  github_repo: Grillcheese-AI/cubemind
  knowledge_docs_path: docs/project_knowledge
redis_ttl: 31536000
redis_url: redis://localhost:6380
relevance_threshold: 0.1
rss_feeds:
- https://hackernoon.com/feed
- https://globalnews.ca/feed/
- https://feedx.net/rss/ap.xml
- https://www.theverge.com/rss/index.xml
- https://feeds.arstechnica.com/arstechnica/index
- https://techcrunch.com/feed/
- https://blog.bytebytego.com/feed
- https://www.wired.com/feed/tag/ai/latest/rss
- https://www.wired.com/feed/category/ideas/latest/rss
- https://rss.arxiv.org/rss/math.QA
- https://rss.arxiv.org/rss/cs.ai
- https://www.reddit.com/r/news/.rss
- https://www.reddit.com/r/LocalLLaMA/.rss
- https://www.reddit.com/r/singularity/.rss
- https://www.cbc.ca/webfeed/rss/rss-topstories
- https://www.cbc.ca/webfeed/rss/rss-technology
- https://www.cbc.ca/webfeed/rss/rss-world
- https://rss.arxiv.org/rss/cs.NE
- https://rss.arxiv.org/rss/cs.LG
- https://rss.arxiv.org/rss/cs.CL
- https://feeds.feedburner.com/venturebeat/SZYF
- https://www.technologyreview.com/feed/
- https://www.reuters.com/rssFeed/worldNews
- https://feeds.bbci.co.uk/news/world/rss.xml
rss_fetch_full_articles: true
rss_max_articles_per_feed: 10
scope_guard: true
skip_dirs:
- node_modules
- .venv
- dist
user_profile:
  auto_observe: true
  decay_days: 90
  enabled: true
vector_search:
  enabled: true
  encoder_model: all-MiniLM-L6-v2
  qdrant_url: http://localhost:6333
---
