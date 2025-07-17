NL2Cypher: Gemini-based Natural Language to Cypher Converter

This project provides an advanced Natural Language to Cypher (NL2Cypher) conversion tool, enabling users to generate Cypher queries for Neo4j databases from plain English descriptions. It leverages Google's Gemini large language model for robust and accurate conversions.
Features

The NL2Cypher application offers the following key capabilities:

    Hybrid Cypher Generation: Combines pattern matching for simple queries, decomposition for complex ones, and dynamic few-shot learning for enhanced accuracy.

    Contextual Learning: Utilizes a ChromaDB vector store to learn from successful queries and logged errors, improving future generations.

    Schema Integration: Processes natural language schema descriptions and ensures generated Cypher queries adhere to your Neo4j database structure.

    Neo4j Integration (Optional): Supports direct connection to Neo4j for real-time query validation and automatic error repair.

    Gradio User Interface: Provides an intuitive web interface for configuration, query input, and displaying results.

Installation

To set up and run the NL2Cypher application, follow these steps:

    Clone the Repository:

    git clone <repository_url>
    cd nl-2-cypher

    (Replace <repository_url> with the actual URL if this project is hosted.)

    Create a Virtual Environment (Recommended):

    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

    Install Dependencies:

    pip install openai chromadb neo4j gradio python-dotenv

    Set up Gemini API Key:

        Obtain your Gemini API Key from Google AI Studio.

        Create a file named .env in the root directory of the project (where app.py is located).

        Add your API key to the .env file in the following format:

        GEMINI_API_KEY="your_gemini_api_key_here"

        Alternatively, you can directly paste your API key into the "Gemini API Key" field in the Gradio UI after launching the application.

Usage

To run the NL2Cypher application:

    Start the Gradio Application:

    python app.py

    This command will launch the Gradio web interface, typically accessible at http://127.0.0.1:7860 in your web browser.

    Interact with the UI:

        Configuration Tab:

            Initialize LLM: Enter your Gemini API Key and click "Initialize LLM". This is a prerequisite for all query generation.

            Set Schema: Provide a natural language description of your Neo4j database schema (e.g., "Users with name, email. Products with id, price. BOUGHT relationships between Users and Products"). Click "Set Schema" to convert it into a structured JSON format.

            Neo4j Connection (Optional): If you wish to enable live validation and auto-repair, enter your Neo4j URI, username, and password, then click "Connect to Neo4j".

        Query Generator Tab:

            Natural Language Query: Enter your natural language question (e.g., "Find all users who bought products costing more than 50 dollars").

            Generate Cypher Query: Click this button to generate the corresponding Cypher query. The generated query, its validation status (if connected to Neo4j), and the generation strategy used will be displayed.

Project Structure

    app.py: The main application file, integrating all core logic and the Gradio UI.

    .env: (Optional) Stores your Gemini API Key.

    chroma_data/: Directory for persistent ChromaDB vector store data.
