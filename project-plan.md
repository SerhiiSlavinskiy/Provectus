Build an agentic system that can: analyze the query to determine needed tools, call a RAG tool to search internal docs, optionally call a web search tool for external info, synthesize a final answer with citations, implement state management for multi-turn conversations, include circuit breakers and error handling. Requirements: 
- Implement all TODO methods using LangGraph with proper state management 
- Use conditional edges for routing 
- Implement circuit breaker pattern (max tool calls) 
- Include checkpointing for conversation persistence - Production-ready error handling 
- Mock the RAG/web search (focus on orchestration)