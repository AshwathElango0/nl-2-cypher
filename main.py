from hybrid_generator import HybridGenerator
from neo4j import GraphDatabase
import openai

# Initialize
llm_client = openai.OpenAI(
    api_key="your-key",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)
generator = HybridGenerator(llm_client)

# Set schema
schema = {
    "nodes": ["User", "Product"],
    "relationships": ["BOUGHT", "REVIEWED"]
}

# Generate queries
print(generator.generate_cypher("Count users who bought products", schema))  # Pattern-matched
print(generator.generate_cypher("Find users who bought and reviewed products", schema))  # Decomposed
print(generator.generate_cypher("Find premium users who bought expensive products", schema))  # Standard