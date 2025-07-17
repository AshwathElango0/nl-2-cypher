from typing import List, Dict
import chromadb
import time

class ContextManager:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.query_collection = self.chroma_client.create_collection("query_examples")
        self.error_log = []
    
    def add_example(self, nl_query: str, cypher: str):
        """Store successful queries with embeddings"""
        self.query_collection.add(
            documents=[cypher],
            metadatas=[{"nl_query": nl_query}],
            ids=[f"id{len(self.query_collection.get()['ids'])}"]
        )
    
    def log_error(self, nl_query: str, error: str):
        """Track errors for contextual learning"""
        self.error_log.append({
            "query": nl_query,
            "error": error,
            "timestamp": time.time()
        })
    
    def get_dynamic_examples(self, nl_query: str, schema: Dict, n: int = 5) -> List[str]:
        """Retrieve balanced context examples"""
        # 1. Semantic matches (3)
        semantic_results = self.query_collection.query(
            query_texts=[nl_query],
            n_results=3
        )['documents'][0]
        
        # 2. Schema-specific examples (1)
        schema_example = self._get_schema_example(schema)
        
        # 3. Recent correction (1)
        correction = self._get_recent_correction()
        
        return semantic_results[:3] + [schema_example] + [correction]
    
    def _get_schema_example(self, schema: Dict) -> str:
        """Find example using rare schema elements"""
        # Implementation simplified for brevity
        return f"MATCH (n:{schema['nodes'][0]}) RETURN n LIMIT 10"
    
    def _get_recent_correction(self) -> str:
        """Get last corrected error if available"""
        return self.error_log[-1]["corrected_query"] if self.error_log else ""