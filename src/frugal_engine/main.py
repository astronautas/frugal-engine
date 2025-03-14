from pyspark.sql import SparkSession
from sqlglot import parse_one, exp
import pandas as pd
import time
import sqlglot
from enum import Enum
import logging
from duckdb import DuckDBPyConnection
from dataclasses import dataclass
import duckdb
from typing import Optional
import polars as pl
from IPython import get_ipython

logger = logging.getLogger(__name__)

class PolarsClient:
    """
    Just an interface similar to Spark and DuckDB i.e. exposing sql, but going through UC catalog
    """
    def __init__(self, workspace_url: str, bearer_token: str | None = None, memory_limit: int | None = None, require_https: bool = False):
        self.unity_catalog_uri = workspace_url
        self.catalog = pl.Catalog(workspace_url=workspace_url, bearer_token=bearer_token, require_https=require_https) # last flag - for local testing
        self.memory_limit = memory_limit

    def sql(self, query: str) -> pl.DataFrame:
        tables = [table for table in sqlglot.parse_one(query).find_all(sqlglot.exp.Table)]

        assert len(tables) == 1, "Only one table is supported as a source for now"

        df_scanned = [self.catalog.scan_table(str(table.catalog), str(table.db), str(table.this)) for table in tables]
        df_scanned = df_scanned[0]

        var_name = next(name for name, val in locals().items() if val is df_scanned)

        query_for_pl = sqlglot.parse_one(query).from_(var_name).sql()
        return pl.sql(query_for_pl).collect()
    
def setup(unity_catalog_uri: str = "http://localhost:8080", threshold: int = 6_000_000):
    spark = _build_local_spark_client(unity_catalog_uri)
    polars_client = PolarsClient(unity_catalog_uri)
    client = HybridClient(spark_client=spark, polars_client=polars_client, threshold=threshold)

    # non-invasive way to add a magic function
    get_ipython().register_magic_function(lambda line, cell=None: client.sql(line or cell),
        magic_kind="line_cell", magic_name="sequel")
    
    return client
                          
def _build_local_spark_client(unity_catalog_uri: str) -> SparkSession:
    spark = SparkSession.builder \
        .appName("local-uc-test") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1,io.unitycatalog:unitycatalog-spark_2.12:0.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.catalog.unity", "io.unitycatalog.spark.UCSingleCatalog") \
        .config("spark.sql.catalog.unity.uri", unity_catalog_uri) \
        .config("spark.sql.catalog.unity.token", "") \
        .config("spark.sql.defaultCatalog", "unity") \
        .config("spark.databricks.delta.catalog.update.enabled", "true") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    return spark

def _build_duckdb_client(unity_catalog_url: str, unity_catalog_token: str, unity_catalog_name) -> DuckDBPyConnection:
    client = duckdb.connect(database=':memory:', read_only=False)

    # client.sql("SET memory_limit = '100KB';")

    client.sql("""
    install uc_catalog from core_nightly;
    load uc_catalog;
    install delta;
    load delta;        
    """)

    client.sql(f"""
    CREATE SECRET (
        TYPE UC,
        TOKEN '{unity_catalog_token}',
        ENDPOINT '{unity_catalog_url}',
        AWS_REGION 'us-east-2'
    );
            
    """)

    client.sql(f"ATTACH '{unity_catalog_name}' AS {unity_catalog_name} (TYPE UC_CATALOG);")

    return client

class Choice(Enum):
    POLARS = "polars"
    SPARK = "spark"

class HybridClient:
    """
    Uses a heuristic - based on the maximum number within the source tables, it decides whether to use DuckDB or Spark.
    Essentially, duckdb acts like a cache, fallbacking to Spark for whatever unforseen exceptions e.g. OOM, unsupported operation, etc
    """
    
    def __init__(self,
                 polars_client: PolarsClient,
                 spark_client: SparkSession,
                 threshold: int):
        
        self.spark_client = spark_client
        self.polars_client = polars_client
        self.threshold = threshold
            
    def _count_heuristic(self, query: str) -> Choice:
        """
        Could be ML-based if we are fancy.
        """

        # query_transpiled = sqlglot.transpile(sql=query, read="spark", write="duckdb")[0]

        tables = [table for table in parse_one(query).find_all(exp.Table)]
        counts = [self.polars_client.sql(f"select count(*) from {table.catalog}.{table.db}.{table.name}").to_series()[0] for table in tables]
        top_count = max(counts)

        logger.debug(f"Top source count: {top_count}")
        
        if top_count <= self.threshold:
            logger.info(f"single_node_executor has been chosen, because the top source count is less than or equal to the threshold: {self.threshold}")
            return Choice.POLARS
        else:
            logger.info(f"cluster_executor has been chosen, because the top source count is greater than the threshold: {self.threshold}")
            return Choice.SPARK

    def sql(self, query: str, engine: Choice = None) -> pd.DataFrame:
        logger.debug(f"Query: {query}")
        
        choice = engine or self._count_heuristic(query)
        logger.debug(f"Choice: {choice}")

        if choice == Choice.POLARS:
            try:
                logger.info(f"Executing query in {Choice.POLARS.value}: {query}")
                # polars - unsupported dialect, duckdb might be closest?
                query = sqlglot.transpile(sql=query, write="duckdb")[0]
                return self.polars_client.sql(query).to_pandas()
            # fixme - too generic exception
            except Exception as e:
                logger.error(f"Error: {e}, fallbacking to {Choice.SPARK.value}")                
                query = sqlglot.transpile(sql=query, write="spark")[0]

            return self.spark_client.sql(query).toPandas()
        else:
            logger.info(f"Executing query in {Choice.SPARK.value}: {query}")
            return self.spark_client.sql(query).toPandas()