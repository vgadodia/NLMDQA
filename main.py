# Natural Language Multi-Database Query Agent (NLMDQA)
# Required packages:
# pip install openai python-dotenv pymongo psycopg2-binary neo4j redis

import os
from typing import Dict, List, Any, Union
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import psycopg2
from pymongo import MongoClient
from neo4j import GraphDatabase
import redis
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Configuration class for database connections"""
    def __init__(self):
        load_dotenv()
        
        # PostgreSQL configuration
        self.pg_config = {
            'dbname': os.getenv('PG_DATABASE'),
            'user': os.getenv('PG_USER'),
            'password': os.getenv('PG_PASSWORD'),
            'host': os.getenv('PG_HOST'),
            'port': os.getenv('PG_PORT')
        }
        
        # MongoDB configuration
        self.mongo_config = {
            'uri': os.getenv('MONGO_URI'),
            'database': os.getenv('MONGO_DATABASE')
        }
        
        # Neo4j configuration
        self.neo4j_config = {
            'uri': os.getenv('NEO4J_URI'),
            'user': os.getenv('NEO4J_USER'),
            'password': os.getenv('NEO4J_PASSWORD')
        }
        
        # Redis configuration
        self.redis_config = {
            'host': os.getenv('REDIS_HOST'),
            'port': os.getenv('REDIS_PORT'),
            'password': os.getenv('REDIS_PASSWORD')
        }
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
class QueryParser:
    """Handles natural language parsing using GPT-4 to generate database queries directly"""
    def __init__(self, config: DatabaseConfig):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        
    async def parse_query(self, natural_language_query: str) -> Dict:
        """Generate database-specific queries directly from natural language"""
        prompt = f"""
        Convert the following natural language query into specific database queries for PostgreSQL, MongoDB, and Neo4j.
        
        Natural Language Query: {natural_language_query}
        
        For context, here is our database schema:
        
        PostgreSQL Tables:
        - products (id, name, price, category)
        - orders (id, customer_id, order_date, total_amount)
        - customers (id, name, email)
        - order_items (order_id, product_id, quantity, price)
        
        MongoDB Collections:
        - reviews (
            _id,
            product_id,
            customer_id,
            rating,
            review_text,
            review_date
        )
        
        Neo4j Structure:
        - Nodes:
            - Customer (id, name)
            - Product (id, name)
            - Order (id, date)
        - Relationships:
            - (Customer)-[PURCHASED]->(Product)
            - (Customer)-[PLACED]->(Order)
            - (Order)-[CONTAINS]->(Product)

        Return ONLY a JSON object with this exact structure:
        {{
            "queries": {{
                "postgresql": "Your PostgreSQL query here",
                "mongodb": {{
                    "collection": "collection_name",
                    "filter": {{}} # MongoDB query filter
                }},
                "neo4j": "Your Cypher query here"
            }},
            "merge_keys": ["customer_id", "product_id"]
        }}
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.choices[0].message.content)

class DatabaseConnector:
    """Manages connections to different databases"""
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_config['host'],
            port=config.redis_config['port'],
            password=config.redis_config['password']
        )
    
    def get_postgres_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.config.pg_config)
    
    def get_mongo_connection(self):
        """Get MongoDB connection"""
        client = MongoClient(self.config.mongo_config['uri'])
        return client[self.config.mongo_config['database']]
    
    def get_neo4j_connection(self):
        """Get Neo4j connection"""
        return GraphDatabase.driver(
            self.config.neo4j_config['uri'],
            auth=(self.config.neo4j_config['user'], 
                  self.config.neo4j_config['password'])
        )

class QueryGenerator:
    """Generates database-specific queries"""
    def generate_postgres_query(self, query_components: Dict) -> str:
        """Generate PostgreSQL query from components"""
        tables = query_components['tables']
        conditions = query_components['conditions']
        joins = query_components['joins']
        
        # Build query using components
        query = f"SELECT * FROM {', '.join(tables)}"
        if joins:
            query += f" {' '.join(joins)}"
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        
        return query
    
    def generate_mongo_query(self, query_components: Dict) -> Dict:
        """Generate MongoDB query from components"""
        return {
            'collection': query_components['collections'][0],
            'filter': {cond['field']: cond['value'] 
                      for cond in query_components['conditions']}
        }
    
    def generate_neo4j_query(self, query_components: Dict) -> str:
        """Generate Neo4j Cypher query from components"""
        nodes = query_components['nodes']
        relationships = query_components['relationships']
        conditions = query_components['conditions']
        
        # Build Cypher query
        query = f"MATCH "
        # Add node patterns
        node_patterns = [f"({node})" for node in nodes]
        query += "-".join(node_patterns)
        
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        query += " RETURN " + ", ".join(nodes)
        
        return query

class QueryExecutor:
    """Executes queries across different databases"""
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def execute_postgres_query(self, query: str) -> List[Dict]:
        """Execute PostgreSQL query"""
        with self.connector.get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                results = []
                for row in cur.fetchall():
                    results.append(dict(zip(columns, row)))
        return results
    
    def execute_mongo_query(self, query: Dict) -> List[Dict]:
        """Execute MongoDB query"""
        db = self.connector.get_mongo_connection()
        collection = db[query['collection']]
        return list(collection.find(query['filter']))
    
    def execute_neo4j_query(self, query: str) -> List[Dict]:
        """Execute Neo4j query"""
        with self.connector.get_neo4j_connection() as driver:
            with driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]

class DataMerger:
    """Merges results from different databases"""
    def merge_results(self, results: Dict[str, List[Dict]], merge_keys: List[str]) -> List[Dict]:
        """Merge results based on common keys"""
        if not results:
            return []
        
        # Start with the first database's results
        merged_data = results[list(results.keys())[0]]
        
        # Merge with other databases
        for db_name in list(results.keys())[1:]:
            merged_data = self._merge_two_datasets(
                merged_data,
                results[db_name],
                merge_keys
            )
        
        return merged_data
    
    def _merge_two_datasets(self, data1: List[Dict], data2: List[Dict], 
                           merge_keys: List[str]) -> List[Dict]:
        """Merge two datasets based on common keys"""
        merged = []
        for item1 in data1:
            for item2 in data2:
                if all(item1.get(key) == item2.get(key) for key in merge_keys):
                    merged.append({**item1, **item2})
        return merged

class CacheManager:
    """Manages query caching using Redis"""
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    def get_cached_result(self, query: str) -> Union[List[Dict], None]:
        """Get cached query result"""
        cached = self.redis_client.get(self._get_cache_key(query))
        return json.loads(cached) if cached else None
    
    def cache_result(self, query: str, result: List[Dict]):
        """Cache query result"""
        self.redis_client.setex(
            self._get_cache_key(query),
            self.cache_ttl,
            json.dumps(result)
        )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return f"nlmdqa:query:{hash(query)}"

class NLMDQA:
    """Main class for Natural Language Multi-Database Query Agent"""
    def __init__(self):
        print("Initializing NLMDQA...")
        self.config = DatabaseConfig()
        self.parser = QueryParser(self.config)
        self.connector = DatabaseConnector(self.config)
        self.query_generator = QueryGenerator()
        self.executor = QueryExecutor(self.connector)
        self.merger = DataMerger()
        self.cache_manager = CacheManager(self.connector.redis_client)

        print("NLMDQA initialized")
    
    async def process_query(self, natural_language_query: str) -> Dict:
        """Process natural language query and return results"""
        # Check cache first
        # cached_result = self.cache_manager.get_cached_result(natural_language_query)
        # if cached_result:
        #     logger.info("Returning cached result")
        #     return cached_result
        
        # # Parse the query manually
        # parsed_query = await self.parser.parse_query(natural_language_query)
        
        # # Generate database-specific queries
        # queries = {
        #     'postgresql': self.query_generator.generate_postgres_query(
        #         parsed_query['query_components']['postgresql']
        #     ),
        #     'mongodb': self.query_generator.generate_mongo_query(
        #         parsed_query['query_components']['mongodb']
        #     ),
        #     'neo4j': self.query_generator.generate_neo4j_query(
        #         parsed_query['query_components']['neo4j']
        #     )
        # }

        # Get queries directly from GPT-4
        query_result = await self.parser.parse_query(natural_language_query)
        print("Generated queries: \n\n", json.dumps(query_result, indent=2))
        
        # Execute queries in parallel
        with ThreadPoolExecutor() as executor:
            results = {
                'postgresql': executor.submit(
                    self.executor.execute_postgres_query, 
                    query_result['queries']['postgresql']
                ),
                'mongodb': executor.submit(
                    self.executor.execute_mongo_query,
                    query_result['queries']['mongodb']
                ),
                'neo4j': executor.submit(
                    self.executor.execute_neo4j_query,
                    query_result['queries']['neo4j']
                )
            }
        
         # Get results from futures
        query_results = {
            db: future.result() 
            for db, future in results.items()
        }
        
        # Merge results
        merged_results = self.merger.merge_results(
            query_results,
            query_result['merge_keys']
        )
        
        # Cache results
        self.cache_manager.cache_result(natural_language_query, merged_results)
        
        return merged_results

# Example usage
async def main():
    # Initialize the NLMDQA system
    nlmdqa = NLMDQA()
    
    # Example query
    query = """Find all customers who purchased products over $100 
               and left reviews with ratings above 4 stars, 
               and show their purchase history"""
    
    try:
        results = await nlmdqa.process_query(query)
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())