import os
import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase
import openai
from typing import List, Optional, Tuple, Dict
import json
import re
import time
import gradio as gr

# Load environment variables from .env file (for API key)
load_dotenv()

# --- ContextManager Class (from context_manager.py) ---
class ContextManager:
    """Manages context examples and error logging for NL2Cypher generation."""
    def __init__(self):
        # Initialize ChromaDB client and collection for query examples
        self.chroma_client = chromadb.Client()
        self.query_collection = self.chroma_client.get_or_create_collection("query_examples")
        self.error_log = [] # Stores errors for potential future correction examples

    def add_example(self, nl_query: str, cypher: str):
        """
        Stores successful natural language query and its corresponding Cypher query
        with embeddings in ChromaDB for future retrieval.
        """
        try:
            self.query_collection.add(
                documents=[cypher],
                metadatas=[{"nl_query": nl_query}],
                ids=[f"id{len(self.query_collection.get()['ids'])}"]
            )
            print(f"Added example: '{nl_query}' -> '{cypher}' to vector store.")
        except Exception as e:
            print(f"Error adding example to ChromaDB: {e}")

    def log_error(self, nl_query: str, error: str, corrected_query: Optional[str] = None):
        """
        Logs an error encountered during query generation or validation.
        Optionally stores a corrected query if available.
        """
        self.error_log.append({
            "query": nl_query,
            "error": error,
            "timestamp": time.time(),
            "corrected_query": corrected_query
        })
        print(f"Logged error for query '{nl_query}': {error}")

    def get_dynamic_examples(self, nl_query: str, schema: Dict, n: int = 5) -> List[str]:
        """
        Retrieves a balanced set of context examples for dynamic few-shot learning.
        Includes semantic matches, a schema-specific example, and a recent correction.
        """
        examples = []

        # 1. Semantic matches (up to 3)
        try:
            semantic_results = self.query_collection.query(
                query_texts=[nl_query],
                n_results=min(n, 3) # Get up to 3 semantic matches
            )
            if semantic_results and semantic_results['documents']:
                for doc, meta in zip(semantic_results['documents'][0], semantic_results['metadatas'][0]):
                    examples.append(f"NL: {meta['nl_query']}\nCypher: {doc}")
        except Exception as e:
            print(f"Error retrieving semantic examples from ChromaDB: {e}")

        # 2. Schema-specific example (1) - simplified for brevity, based on first node label
        if schema and schema.get("nodes"):
            schema_example = self._get_schema_example(schema)
            if schema_example:
                examples.append(f"NL: Example for schema node\nCypher: {schema_example}")

        # 3. Recent correction (1)
        correction = self._get_recent_correction()
        if correction:
            examples.append(f"NL: Recent corrected query\nCypher: {correction}")
        
        # Ensure unique examples and limit to n
        return list(dict.fromkeys(examples))[:n]

    def _get_schema_example(self, schema: Dict) -> str:
        """
        Generates a simple Cypher example based on the provided schema.
        Currently, it returns a basic query for the first node label.
        """
        if schema and schema.get("nodes") and len(schema["nodes"]) > 0:
            # Assuming nodes is a list of strings (labels) or dicts with 'label' key
            first_node_label = schema["nodes"][0]
            if isinstance(first_node_label, dict) and "label" in first_node_label:
                first_node_label = first_node_label["label"]
            elif not isinstance(first_node_label, str):
                first_node_label = "Node" # Default if unexpected format
            return f"MATCH (n:{first_node_label}) RETURN n LIMIT 5"
        return ""

    def _get_recent_correction(self) -> str:
        """
        Retrieves the Cypher query from the most recent error that had a correction.
        """
        for entry in reversed(self.error_log):
            if entry.get("corrected_query"):
                return entry["corrected_query"]
        return ""

# --- QueryDecomposer Class (from query_decomposer.py) ---
class QueryDecomposer:
    """Decomposes complex natural language queries into simpler steps and combines Cypher results."""
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate_steps(self, nl_query: str) -> List[str]:
        """
        Breaks down a complex natural language query into atomic, manageable steps
        using the LLM.
        """
        try:
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
            content = response.choices[0].message.content
            # Ensure content is a string before json.loads
            if isinstance(content, str):
                return json.loads(content)["steps"]
            else:
                print(f"Warning: LLM response content is not a string: {content}")
                return []
        except Exception as e:
            print(f"Error generating decomposition steps: {e}")
            return []

    def combine_results(self, partial_queries: List[str]) -> str:
        """
        Combines multiple partial Cypher queries into a single, optimized Cypher query
        using the LLM.
        """
        try:
            response = self.llm.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{
                    "role": "system",
                    "content": "Combine these into one optimized Cypher query. Ensure the combined query is syntactically correct and logically sound."
                }, {
                    "role": "user",
                    "content": "\n".join(partial_queries)
                }],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error combining partial queries: {e}")
            return "Error: Could not combine queries."

# --- HybridGenerator Class (from hybrid_generator.py) ---
class HybridGenerator:
    """
    Generates Cypher queries using a hybrid approach:
    1. Deterministic pattern matching
    2. Step-back decomposition for complex queries
    3. Standard LLM generation with dynamic few-shot examples
    """
    def __init__(self, llm_client, context_manager: ContextManager, query_decomposer: QueryDecomposer):
        self.llm = llm_client
        self.context = context_manager
        self.decomposer = query_decomposer
        # Pre-defined patterns for quick and accurate generation
        self.patterns = [
            (r"count\s+.*nodes", "MATCH (n:{label}) RETURN COUNT(n)"),
            (r"find\s+.*connected\s+to", "MATCH (a)-[r]->(b) WHERE id(a) = {id_a} AND id(b) = {id_b} RETURN a, r, b"), # Placeholder for specific connection
            (r"find\s+.*nodes\s+with\s+property", "MATCH (n:{label}) WHERE n.{property} = '{value}' RETURN n"),
            (r"match\s+all\s+nodes", "MATCH (n) RETURN n LIMIT 25")
        ]
    
    def generate_cypher(self, nl_query: str, schema: Dict) -> Tuple[str, Optional[str]]:
        """
        Generates a Cypher query based on the natural language query and schema.
        Returns the Cypher query and the generation strategy used.
        """
        # 1. First try deterministic patterns
        for pattern_re, template in self.patterns:
            if re.search(pattern_re, nl_query.lower()):
                cypher = self._apply_pattern(template, nl_query, schema)
                if cypher:
                    return cypher, "Pattern-matched"
        
        # 2. For complex queries, use decomposition
        if self._is_complex(nl_query):
            return self._generate_complex(nl_query, schema)
        
        # 3. Standard generation with dynamic few-shot
        return self._generate_standard(nl_query, schema)
    
    def _apply_pattern(self, template: str, nl_query: str, schema: Dict) -> str:
        """
        Fills pattern templates with schema elements or extracted values from the NL query.
        This is a simplified implementation and might need more sophisticated parsing
        for real-world scenarios.
        """
        # Replace {label} with the first node label from the schema if available
        if "{label}" in template and schema and schema.get("nodes") and len(schema["nodes"]) > 0:
            first_node_label = schema["nodes"][0]
            if isinstance(first_node_label, dict) and "label" in first_node_label:
                template = template.replace("{label}", first_node_label["label"])
            elif isinstance(first_node_label, str):
                template = template.replace("{label}", first_node_label)
            else:
                template = template.replace("{label}", "Node") # Fallback
        
        # Simple extraction for property/value (can be improved with regex or NER)
        if "{property}" in template and "{value}" in template:
            # Example: "find users with name 'Alice'"
            match = re.search(r"with\s+(\w+)\s+'([^']+)'", nl_query, re.IGNORECASE)
            if match:
                prop, val = match.groups()
                template = template.replace("{property}", prop).replace("{value}", val)
            else:
                template = template.replace("WHERE n.{property} = '{value}'", "") # Remove if no match
        
        # Placeholder for {relationship}
        if "{relationship}" in template:
            # This would ideally be extracted from NL or inferred from schema
            template = template.replace("{relationship}", "REL") # Default placeholder
        
        # Placeholder for {id_a} and {id_b}
        if "{id_a}" in template and "{id_b}" in template:
            # This would require more advanced entity extraction
            template = template.replace("{id_a}", "1").replace("{id_b}", "2") # Default placeholder
            
        return template

    def _is_complex(self, query: str) -> bool:
        """
        Heuristic to determine if a query is complex enough to warrant decomposition.
        Checks for length and common complex keywords.
        """
        return len(query.split()) > 8 or " and " in query.lower() or " or " in query.lower() or " not " in query.lower()

    def _generate_complex(self, nl_query: str, schema: Dict) -> Tuple[str, Optional[str]]:
        """
        Generates Cypher for complex queries using a step-back decomposition approach.
        """
        steps = self.decomposer.generate_steps(nl_query)
        if not steps:
            return "Error: Could not decompose query.", "Decomposition Failed"

        partials = []
        for step in steps:
            # Recursively call standard generation for each step
            cypher, _ = self._generate_standard(step, schema)
            partials.append(cypher)
        
        combined_cypher = self.decomposer.combine_results(partials)
        return combined_cypher, "Decomposed"
    
    def _generate_standard(self, nl_query: str, schema: Dict) -> Tuple[str, Optional[str]]:
        """
        Generates Cypher using standard LLM call with dynamic few-shot examples.
        """
        examples = self.context.get_dynamic_examples(nl_query, schema)
        
        # Format examples for the prompt
        formatted_examples = "\n".join([f"NL: {ex.split('NL: ')[1].split('\nCypher: ')[0]}\nCypher: {ex.split('\nCypher: ')[1]}" for ex in examples if "NL:" in ex and "Cypher:" in ex])

        prompt = f"""
        You are an expert in Cypher query language for Neo4j.
        Generate a Cypher query for the given natural language query based on the provided schema.
        Ensure the query is syntactically correct and uses only the labels and relationship types from the schema.

        Schema: {json.dumps(schema, indent=2)}

        Examples:
        {formatted_examples}

        Natural Language Query: {nl_query}

        Cypher Query:
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip(), "Standard LLM"
        except Exception as e:
            print(f"Error during standard LLM generation: {e}")
            return f"Error: LLM generation failed - {e}", "LLM Error"

# --- Main NL2CypherAdvanced Class (Integrated) ---
class NL2CypherAdvanced:
    """
    Comprehensive Natural Language to Cypher engine with schema understanding,
    hybrid generation, Neo4j integration, and feedback mechanisms.
    """
    def __init__(self):
        self.driver = None # Neo4j driver
        self.schema = None # Stores the parsed database schema
        self.llm_client = None # LLM client (e.g., OpenAI/Gemini)
        self.context_manager = ContextManager() # Manages examples and errors
        self.query_decomposer = None # Decomposes complex queries
        self.hybrid_generator = None # Generates Cypher using hybrid strategy

    def initialize_llm(self, api_key: str):
        """Initializes the LLM client."""
        try:
            self.llm_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/"
            )
            self.query_decomposer = QueryDecomposer(self.llm_client)
            self.hybrid_generator = HybridGenerator(self.llm_client, self.context_manager, self.query_decomposer)
            print("LLM client initialized successfully.")
            return "LLM client initialized successfully."
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            return f"Error initializing LLM client: {e}"

    def set_schema(self, schema_description: str) -> Tuple[Optional[Dict], str]:
        """
        Converts a natural language schema description into a structured JSON format
        using the LLM.
        """
        if not self.llm_client:
            return None, "Error: LLM client not initialized. Please provide an API key."

        prompt = f"""
        Convert the following natural language description of a Neo4j database schema into a JSON object.
        The JSON should have two top-level keys: "nodes" and "relationships".
        
        For "nodes", provide a list of node labels. If properties are mentioned, include them as a dictionary for each node, e.g., {{"label": "User", "properties": ["name", "email"]}}.
        For "relationships", provide a list of relationship types. If source and target nodes are mentioned, include them as a dictionary, e.g., {{"type": "BOUGHT", "from": "User", "to": "Product"}}.

        Natural Language Schema Description:
        {schema_description}

        JSON Schema:
        """
        try:
            response = self.llm_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                self.schema = json.loads(content)
                print(f"Schema set: {json.dumps(self.schema, indent=2)}")
                return self.schema, "Schema set successfully."
            else:
                return None, f"Error: LLM response content is not a string for schema conversion: {content}"
        except Exception as e:
            self.schema = None
            print(f"Error setting schema: {e}")
            return None, f"Error setting schema: {e}"

    def connect_neo4j(self, uri: str, user: str, pwd: str) -> str:
        """
        Connects to a Neo4j database and validates the connection.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            with self.driver.session() as session:
                session.run("RETURN 1").consume() # Test connection
            print("Successfully connected to Neo4j.")
            return "Successfully connected to Neo4j."
        except Exception as e:
            self.driver = None
            print(f"Failed to connect to Neo4j: {e}")
            return f"Failed to connect to Neo4j: {e}"

    def generate_and_validate_cypher(self, nl_query: str) -> Tuple[str, str, str]:
        """
        Generates a Cypher query, optionally validates it against Neo4j,
        and attempts auto-repair if validation fails.
        Returns the generated Cypher, validation status, and generation strategy.
        """
        if not self.llm_client or not self.hybrid_generator:
            return "", "Error: LLM client or Hybrid Generator not initialized. Please provide an API key.", "N/A"
        if not self.schema:
            return "", "Error: Schema not set. Please set the schema first.", "N/A"

        cypher, strategy = self.hybrid_generator.generate_cypher(nl_query, self.schema)
        validation_status = "Not validated (No Neo4j connection)"

        if self.driver:
            try:
                with self.driver.session() as session:
                    session.run(f"EXPLAIN {cypher}").consume() # Use EXPLAIN for validation
                validation_status = "Valid"
                self.context_manager.add_example(nl_query, cypher) # Store successful query
            except Exception as e:
                validation_status = f"Invalid: {str(e)}"
                print(f"Validation failed for query: {cypher}\nError: {e}")
                
                # Attempt auto-repair
                repaired_cypher = self._attempt_repair(nl_query, cypher, str(e))
                if repaired_cypher and repaired_cypher != cypher:
                    cypher = repaired_cypher
                    validation_status += " (Attempted Repair)"
                    try:
                        with self.driver.session() as session:
                            session.run(f"EXPLAIN {cypher}").consume()
                        validation_status = "Valid (Repaired)"
                        self.context_manager.add_example(nl_query, cypher) # Store repaired query
                    except Exception as repair_e:
                        validation_status = f"Invalid (Repair Failed): {str(repair_e)}"
                        self.context_manager.log_error(nl_query, str(repair_e), repaired_cypher)
                else:
                    self.context_manager.log_error(nl_query, str(e)) # Log original error if repair not possible/different
        
        return cypher, validation_status, strategy

    def _attempt_repair(self, nl_query: str, faulty_cypher: str, error_message: str) -> Optional[str]:
        """
        Attempts to repair a faulty Cypher query using the LLM, given the error message.
        """
        if not self.llm_client or not self.schema:
            return None

        prompt = f"""
        The following Cypher query failed validation with the error:
        Error: {error_message}

        Faulty Cypher Query:
        {faulty_cypher}

        Natural Language Query (original intent):
        {nl_query}

        Schema: {json.dumps(self.schema, indent=2)}

        Please provide the corrected Cypher query. Respond ONLY with the corrected Cypher query.
        """
        try:
            response = self.llm_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Lower temperature for more deterministic repair
                max_tokens=500
            )
            repaired = response.choices[0].message.content.strip()
            print(f"Attempted repair. Original: '{faulty_cypher}', Repaired: '{repaired}'")
            return repaired
        except Exception as e:
            print(f"Error during auto-repair: {e}")
            return None

    def close(self):
        """Closes the Neo4j driver and cleans up ChromaDB resources."""
        if self.driver:
            self.driver.close()
            print("Neo4j driver closed.")
        # self.context_manager.chroma_client.delete_collection("query_examples") # Only delete if truly done with it
        print("Application resources cleaned up.")

# --- Gradio UI Setup ---

# Instantiate the main NL2Cypher engine
nl2cypher_engine = NL2CypherAdvanced()

def initialize_app(api_key: str):
    """Initializes the LLM client for the application."""
    if not api_key:
        return "Please enter your Gemini API Key.", "", ""
    return nl2cypher_engine.initialize_llm(api_key), "", ""

def set_schema_ui(schema_description: str):
    """Sets the database schema."""
    schema_obj, message = nl2cypher_engine.set_schema(schema_description)
    if schema_obj:
        return message, json.dumps(schema_obj, indent=2)
    return message, "Schema not set or error occurred."

def connect_neo4j_ui(uri: str, user: str, password: str):
    """Connects to Neo4j."""
    return nl2cypher_engine.connect_neo4j(uri, user, password)

def generate_query_ui(nl_query: str):
    """Generates and validates the Cypher query."""
    cypher, validation_status, strategy = nl2cypher_engine.generate_and_validate_cypher(nl_query)
    return cypher, validation_status, strategy

# Gradio Interface
with gr.Blocks(title="NL2Cypher Advanced") as demo:
    gr.Markdown(
        """
        # 🧠 NL2Cypher Advanced: Natural Language to Cypher Query Generator
        Convert your natural language questions into executable Cypher queries for Neo4j.
        This application uses a hybrid approach (pattern matching, query decomposition, and dynamic few-shot learning)
        and can optionally connect to your Neo4j database for real-time validation and auto-repair.
        """
    )

    with gr.Tab("Configuration"):
        with gr.Row():
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="Enter your Gemini API Key here (e.g., AIza...)",
                value=os.getenv("GEMINI_API_KEY", "") # Pre-fill if available from .env
            )
            init_llm_btn = gr.Button("Initialize LLM")
        llm_status_output = gr.Textbox(label="LLM Initialization Status", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            schema_desc_input = gr.Textbox(
                label="Neo4j Schema Description (Natural Language)",
                lines=5,
                placeholder="e.g., Users with name, email, signup_date. Products with id, name, price. ORDERS relationships between Users and Products.",
                value="Users with name, email, signup_date. Products with id, name, price. ORDERS relationships between Users and Products."
            )
            set_schema_btn = gr.Button("Set Schema")
        schema_status_output = gr.Textbox(label="Schema Setting Status", interactive=False)
        parsed_schema_output = gr.Json(label="Parsed Schema (JSON)", interactive=False)
        
        gr.Markdown("---")

        with gr.Accordion("Optional: Neo4j Connection (for Validation & Repair)", open=False):
            with gr.Row():
                neo4j_uri_input = gr.Textbox(label="Neo4j URI", placeholder="bolt://localhost:7687", value="bolt://localhost:7687")
                neo4j_user_input = gr.Textbox(label="Neo4j User", placeholder="neo4j", value="neo4j")
                neo4j_pwd_input = gr.Textbox(label="Neo4j Password", type="password", placeholder="password", value="password")
            connect_neo4j_btn = gr.Button("Connect to Neo4j")
            neo4j_status_output = gr.Textbox(label="Neo4j Connection Status", interactive=False)
    
    with gr.Tab("Query Generator"):
        nl_query_input = gr.Textbox(
            label="Natural Language Query",
            lines=3,
            placeholder="e.g., Find users who bought expensive products (> $100) and reviewed them.",
            value="Find users who bought expensive products (> $100) and reviewed them."
        )
        generate_btn = gr.Button("Generate Cypher Query")

        gr.Markdown("---")

        cypher_output = gr.Code(label="Generated Cypher Query", language="cypher", interactive=False)
        validation_output = gr.Textbox(label="Validation Status", interactive=False)
        strategy_output = gr.Textbox(label="Generation Strategy", interactive=False)

    # Event Handlers
    init_llm_btn.click(
        fn=initialize_app,
        inputs=[api_key_input],
        outputs=[llm_status_output, schema_status_output, neo4j_status_output] # Clear other status on LLM init
    )
    set_schema_btn.click(
        fn=set_schema_ui,
        inputs=[schema_desc_input],
        outputs=[schema_status_output, parsed_schema_output]
    )
    connect_neo4j_btn.click(
        fn=connect_neo4j_ui,
        inputs=[neo4j_uri_input, neo4j_user_input, neo4j_pwd_input],
        outputs=[neo4j_status_output]
    )
    generate_btn.click(
        fn=generate_query_ui,
        inputs=[nl_query_input],
        outputs=[cypher_output, validation_output, strategy_output]
    )

# To run the Gradio app
if __name__ == "__main__":
    # Ensure ChromaDB directory exists if not in-memory
    os.makedirs("./chroma_data", exist_ok=True)
    demo.launch()
