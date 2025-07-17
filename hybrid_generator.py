from typing import Optional, Tuple
from .context_manager import ContextManager
from .query_decomposer import QueryDecomposer
import re
import json
from typing import Dict

class HybridGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.context = ContextManager()
        self.decomposer = QueryDecomposer(llm_client)
        self.patterns = [
            (r"count.*nodes", "MATCH (n:{label}) RETURN COUNT(n)"),
            (r"find.*connected to", "MATCH (a)-[:{relationship}]->(b) RETURN a, b")
        ]
    
    def generate_cypher(self, nl_query: str, schema: Dict) -> Tuple[str, Optional[str]]:
        # First try deterministic patterns
        for pattern_re, template in self.patterns:
            if re.search(pattern_re, nl_query.lower()):
                cypher = self._apply_pattern(template, schema)
                if cypher:
                    return cypher, "Pattern-matched"
        
        # For complex queries, use decomposition
        if self._is_complex(nl_query):
            return self._generate_complex(nl_query, schema)
        
        # Standard generation with dynamic few-shot
        return self._generate_standard(nl_query, schema)
    
    def _apply_pattern(self, template: str, schema: Dict) -> str:
        """Fill pattern templates with schema elements"""
        # Simplified implementation
        return template.replace("{label}", schema["nodes"][0])
    
    def _is_complex(self, query: str) -> bool:
        """Heuristic for query complexity"""
        return len(query.split()) > 10 or " and " in query.lower()
    
    def _generate_complex(self, nl_query: str, schema: Dict) -> Tuple[str, Optional[str]]:
        """Step-back decomposition approach"""
        steps = self.decomposer.generate_steps(nl_query)
        partials = []
        for step in steps:
            cypher, _ = self._generate_standard(step, schema)
            partials.append(cypher)
        return self.decomposer.combine_results(partials), "Decomposed"
    
    def _generate_standard(self, nl_query: str, schema: Dict) -> Tuple[str, Optional[str]]:
        """Standard generation with dynamic few-shot"""
        examples = self.context.get_dynamic_examples(nl_query, schema)
        prompt = f"""
        Schema: {json.dumps(schema)}
        Examples:
        {examples}
        Query: {nl_query}
        """
        response = self.llm.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip(), None