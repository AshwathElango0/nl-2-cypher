from typing import List, Dict
import json

class QueryDecomposer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_steps(self, nl_query: str) -> List[str]:
        """Break complex query into atomic steps"""
        response = self.llm.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{
                "role": "system",
                "content": "Break queries into atomic steps. Respond with JSON: {'steps': [str]}"
            }, {
                "role": "user",
                "content": nl_query
            }],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)["steps"]
    
    def combine_results(self, partial_queries: List[str]) -> str:
        """Merge partial Cypher queries"""
        response = self.llm.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{
                "role": "system",
                "content": "Combine these into one optimized Cypher query:"
            }, {
                "role": "user",
                "content": "\n".join(partial_queries)
            }],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()