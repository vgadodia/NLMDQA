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
import decimal
from json import JSONEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder to handle Decimal types"""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super().default(obj)

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
        try:
            """Generate database-specific queries directly from natural language"""
            prompt = f"""
            Convert the following natural language query into a pipeline of database queries.
            You can create multi-stage queries where results from one query feed into another.
            
            Natural Language Query: {natural_language_query}
            
            IMPORTANT RULES FOR QUERY GENERATION:
            1. Use proper SQL JOINs and CTEs (Common Table Expressions) whenever possible instead of separate queries
            2. For aggregations and statistics, prefer using SQL GROUP BY, HAVING, and window functions
            3. Minimize the number of pipeline stages by combining operations where possible
            4. When querying across tables, maintain proper relationships through JOINs
            5. Use subqueries or CTEs instead of trying to process result sets outside of SQL
            6. Never try to directly substitute array results into a FROM clause
            
            Database Schema (use EXACT column names):
            
            PostgreSQL Tables:
            movies (
                id INTEGER PRIMARY KEY,  # Use 'id', never 'movie_id'
                title TEXT,
                release_year INTEGER,
                certificate TEXT,
                runtime INTEGER,
                imdb_rating FLOAT,
                meta_score INTEGER,
                overview TEXT,
                gross BIGINT,
                no_of_votes INTEGER,
                poster_link TEXT
            )
            
            genres (
                id INTEGER PRIMARY KEY,  # Use 'id', never 'genre_id'
                name TEXT
            )
            
            movie_genres (
                movie_id INTEGER,  # References movies.id
                genre_id INTEGER   # References genres.id
            )
            
            Neo4j Structure:
            Nodes:
            - Movie (id, title)
            - Person (id, name)
            
            Relationships:
            - (Person)-[DIRECTED]->(Movie)
            - (Person)-[ACTED_IN]->(Movie)

            VALIDATION RULES:
            1. Always use 'id' (not 'movie_id') when referring to movies.id
            2. Always use 'id' (not 'genre_id') when referring to genres.id
            3. In movie_genres table, use 'movie_id' and 'genre_id' as they are junction table columns
            4. Double-check all column names against the schema before using them

            IMPORTANT RULES FOR NEO4J QUERIES:
            1. Always use different variable names for relationships and nodes
            2. Never reuse the same variable name in a pattern
            3. Example pattern: (director:Person)-[r:DIRECTED]->(m:Movie)<-[acted:ACTED_IN]-(actor:Person)
            4. Bad pattern: (p:Person)-[a:ACTED_IN]->(m:Movie)<-[a:DIRECTED]-(a:Person)  // Never do this!

            QUERY EXAMPLES:
            
            1. For finding most common genres among high-grossing movies:
            {{
                "pipeline": [
                    {{
                        "stage": 1,
                        "database": "postgresql",
                        "query": {{
                            "postgresql": "
                                WITH high_grossing AS (
                                    SELECT id 
                                    FROM movies 
                                    WHERE gross > 100000000
                                ),
                                genre_counts AS (
                                    SELECT g.name, COUNT(*) as count
                                    FROM genres g
                                    JOIN movie_genres mg ON g.id = mg.genre_id
                                    JOIN high_grossing hg ON mg.movie_id = hg.id
                                    GROUP BY g.name
                                    ORDER BY count DESC
                                )
                                SELECT name, count 
                                FROM genre_counts"
                        }},
                        "output_keys": ["name", "count"],
                        "description": "Get genre counts for high-grossing movies using CTEs"
                    }}
                ],
                "final_merge_keys": ["name"]
            }}
            
            2. For finding movies by director with ratings:
            {{
                "pipeline": [
                    {{
                        "stage": 1,
                        "database": "neo4j",
                        "query": {{
                            "neo4j": "
                                MATCH (p:Person)-[r:DIRECTED]->(m:Movie)
                                WHERE p.name = 'Christopher Nolan'
                                RETURN m.id as id"
                        }},
                        "output_keys": ["id"],
                        "description": "Get movies by director"
                    }},
                    {{
                        "stage": 2,
                        "database": "postgresql",
                        "query": {{
                            "postgresql": "
                                SELECT m.title, m.imdb_rating, m.release_year
                                FROM movies m
                                WHERE m.id IN ({{previous_stage1.id}})
                                ORDER BY m.imdb_rating DESC"
                        }},
                        "output_keys": ["title", "imdb_rating", "release_year"],
                        "description": "Get movie details with ratings"
                    }}
                ],
                "final_merge_keys": ["id"]
            }}

            Return ONLY a JSON object with the exact structure shown in the examples.

            Notes:
            - Each stage can use results from previous stages using placeholder {{previous_stageN.key}} where N is the stage number (e.g. {{previous_stage1.id}})
            - The output_keys specify which fields to pass to the next stage
            - The output_keys specify which fields to pass to the next stage
            - Include a description for each stage explaining its purpose
            - Consider using Neo4j for relationship-heavy queries (directors, actors)
            - Use PostgreSQL for numerical/text filtering and sorting (ratings, gross, years)
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            # return json.loads(response.choices[0].message.content)

            content = response.choices[0].message.content.strip()

            # Clean up the content before parsing JSON
            content = content.replace('\n', ' ')  # Replace newlines with spaces
            content = ' '.join(content.split())   # Normalize whitespace
            
            # Remove any special characters that might interfere with JSON parsing
            content = content.replace('\t', ' ')  # Replace tabs
            
            # If the SQL queries contain newlines in the response, escape them properly
            content = content.replace('": "', '": "')  # Ensure consistent quote formatting
        
            # Add error handling and logging for the JSON parsing
            try:
                return json.loads(content)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response: {content}")
                logger.error(f"JSON parse error: {str(json_err)}")
                raise RuntimeError("Generated query was not valid JSON") from json_err
        except Exception as e:
            logger.error(f"Error during query parsing: {str(e)}")
            raise RuntimeError(f"Failed to parse natural language query: {str(e)}") from e

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
    
    # def execute_postgres_query(self, query: str) -> List[Dict]:
    #     """Execute PostgreSQL query"""
    #     with self.connector.get_postgres_connection() as conn:
    #         with conn.cursor() as cur:
    #             cur.execute(query)
    #             columns = [desc[0] for desc in cur.description]
    #             results = []
    #             for row in cur.fetchall():
    #                 results.append(dict(zip(columns, row)))
    #     return results
    def execute_postgres_query(self, query: str) -> List[Dict]:
        """Execute PostgreSQL query"""
        with self.connector.get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                results = []
                for row in cur.fetchall():
                    # Convert Decimal objects to float
                    processed_row = []
                    for value in row:
                        if isinstance(value, decimal.Decimal):
                            processed_row.append(float(value))
                        else:
                            processed_row.append(value)
                    results.append(dict(zip(columns, processed_row)))
        return results
    
    def execute_mongo_query(self, query: Dict) -> List[Dict]:
        """Execute MongoDB query"""
        db = self.connector.get_mongo_connection()
        collection = db[query['collection']]
        return list(collection.find(query['filter']))
    
    def execute_neo4j_query(self, query: str) -> List[Dict]:
        """Execute Neo4j query"""
        with self.connector.get_neo4j_connection() as driver:
            with driver.session(database="movies-db") as session:
                result = session.run(query)
                return [record.data() for record in result]

class DataMerger:
    """Merges results from different databases"""
    def merge_results(self, results: Dict[str, List[Dict]], merge_keys: List[str]) -> List[Dict]:
        """Merge results based on common keys"""
        if not results:
            return []
        
        # Validate that all merge keys are present in the results
        for stage_name, stage_results in results.items():
            if stage_results:  # Check if there are any results
                missing_keys = [key for key in merge_keys if key not in stage_results[0]]
                if missing_keys:
                    logger.warning(f"Stage {stage_name} is missing merge keys: {missing_keys}")
                    return []  # Return empty list if merge keys are missing
        
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
        
        # Create dictionaries for faster lookup
        data2_dict = {
            tuple(str(item.get(key)) for key in merge_keys): item 
            for item in data2
        }
        
        for item1 in data1:
            key_tuple = tuple(str(item1.get(key)) for key in merge_keys)
            if key_tuple in data2_dict:
                merged_item = {**item1, **data2_dict[key_tuple]}
                merged.append(merged_item)
        
        return merged
    
    # def _merge_two_datasets(self, data1: List[Dict], data2: List[Dict], 
    #                        merge_keys: List[str]) -> List[Dict]:
    #     """Merge two datasets based on common keys"""
    #     merged = []
    #     for item1 in data1:
    #         for item2 in data2:
    #             if all(item1.get(key) == item2.get(key) for key in merge_keys):
    #                 merged.append({**item1, **item2})
    #     return merged

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
        self.config = DatabaseConfig()
        self.parser = QueryParser(self.config)
        self.connector = DatabaseConnector(self.config)
        self.query_generator = QueryGenerator()
        self.executor = QueryExecutor(self.connector)
        self.merger = DataMerger()
        self.cache_manager = CacheManager(self.connector.redis_client)
    
    async def process_query(self, natural_language_query: str) -> Dict:
        """Process natural language query and return results"""
        # Check cache first
        # cached_result = self.cache_manager.get_cached_result(natural_language_query)
        # if cached_result:
        #     logger.info("Returning cached result")
        #     return cached_result

        # Get queries directly from GPT-4
        pipeline_result = await self.parser.parse_query(natural_language_query)
        print("Generated pipeline: \n\n", json.dumps(pipeline_result, indent=2))

        # Execute pipeline stages
        stage_results = {}
        final_results = {}
        test_final_results = None
        
        for stage in pipeline_result['pipeline']:
            stage_num = stage['stage']
            database = stage['database']
            query = stage['query'][database]

            print(f"Stage {stage_num} original query: {query}")
            
            # Replace placeholders with previous stage results
            if isinstance(query, str):
                for prev_stage, prev_results in stage_results.items():
                    for key, value in prev_results.items():
                        # Format list values for SQL IN clauses
                        if isinstance(value, list):
                            formatted_values = []
                            for v in value:
                                if isinstance(v, (int, float)):
                                    formatted_values.append(str(v))
                                else:
                                    formatted_values.append(f"'{v}'")
                            formatted_list = ', '.join(formatted_values)
                            query = query.replace(f"{{previous_stage{prev_stage}.{key}}}", formatted_list)
                        else:
                            query = query.replace(f"{{previous_stage{prev_stage}.{key}}}", str(value))
            elif isinstance(query, dict):  # MongoDB query
                query_str = json.dumps(query)
                for prev_stage, prev_results in stage_results.items():
                    for key, value in prev_results.items():
                        # Format list values for SQL IN clauses
                        if isinstance(value, list):
                            formatted_values = []
                            for v in value:
                                if isinstance(v, (int, float)):
                                    formatted_values.append(str(v))
                                else:
                                    formatted_values.append(f"'{v}'")
                            formatted_list = ', '.join(formatted_values)
                            query_str = query_str.replace(f"{{previous_stage{prev_stage}.{key}}}", formatted_list)
                        else:
                            query_str = query_str.replace(f"{{previous_stage{prev_stage}.{key}}}", str(value))
                query = json.loads(query_str)

            print(f"Stage {stage_num} formatted query: {query}")
            
            # Execute query based on database type
            if database == 'postgresql':
                results = self.executor.execute_postgres_query(query)
            elif database == 'mongodb':
                results = self.executor.execute_mongo_query(query)
            elif database == 'neo4j':
                results = self.executor.execute_neo4j_query(query)
            
            # Store results for this stage
            stage_results[stage_num] = {
                key: [r[key] for r in results] for key in stage['output_keys']
            }
            final_results[f"stage_{stage_num}"] = results
            test_final_results = results

            # print(f"Stage {stage_num} results: {results}")
        
        # Merge results from all stages
        # merged_results = self.merger.merge_results(
        #     final_results,
        #     pipeline_result['final_merge_keys']
        # )
        
        # Cache results
        # self.cache_manager.cache_result(natural_language_query, merged_results)
        
        return test_final_results

# Example usage
async def main():
    # Initialize the NLMDQA system
    nlmdqa = NLMDQA()
    
    # Example query
    query = "Who are the directors that have made the most movies with Robert De Niro?"
    
    try:
        results = await nlmdqa.process_query(query)
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

"""
1. Simple Queries (Single Database)
"Show me all movies with an IMDB rating above 8.5"
"What are the top 10 highest-grossing movies?"
"List all movies released in 2019"

2. Genre-Based Queries (PostgreSQL + Relationships)
"Find all action movies with an IMDB rating above 8.0"
"What are the most common genres among movies that grossed over $100 million?"
"Show me all drama movies from the last 5 years with high meta scores"

3. Person-Related Queries (Neo4j Relationships)
"Which actors have worked with Christopher Nolan?"
"Find all movies where Tom Hanks and Leonardo DiCaprio acted together"
"Who are the directors that have made the most movies with Robert De Niro?"

4. Complex Multi-Database Queries
"Find all war movies directed by Steven Spielberg that grossed over $200 million"
"Show me all movies where Morgan Freeman acted that have an IMDB rating above 8.0 and are in the drama genre"
"List the top 5 directors who have made the highest-grossing sci-fi movies in the last decade"

5. Analytics-Focused Queries
"What's the average IMDB rating for movies directed by Martin Scorsese?"
"Compare the average gross earnings of action movies vs. drama movies"
"Who are the actors that appear most frequently in movies with meta scores above 80?"

6. Time-Based Analysis
"Show the trend of superhero movie ratings over the last 20 years"
"Which directors have consistently made high-grossing movies in each decade?"
"Find movies from the 1990s that have both high IMDB ratings and high meta scores"

7. Complex Relationship Chains
"Find actors who have both directed and acted in movies with ratings above 8.0"
"Show me directors who have worked with the same actor in more than 3 movies"
"List movies where the director has also acted in another director's film"
"""