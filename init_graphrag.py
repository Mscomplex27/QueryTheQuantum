import os
import json
from pyTigerGraph import TigerGraphConnection

def create_graph(conn: TigerGraphConnection):
    conn.gsql(f"""CREATE GRAPH {conn.graphname}()""")
    conn.ai.initializeGraphRAG()

def load_data(conn: TigerGraphConnection):
    res = conn.ai.createDocumentIngest(
        data_source="local",
        data_source_config={"data_path": "./data/tg_tutorials.json"},
        file_format="json",
    )
    conn.ai.runDocumentIngest(res["load_job_id"], res["data_source_id"], res["data_path"])

def update_graphrag(conn: TigerGraphConnection):
    conn.ai.forceConsistencyUpdate("graphrag")

if __name__ == "__main__":

    # We first create a connection to the database
    conn = TigerGraphConnection(
        host="http://localhost",
        username="tigergraph",
        password="tigergraph",
        restppPort="14240",
    )
    conn.graphname = "TigerGraphRAG"

    conn.ai.configureGraphRAGHost("http://localhost:8000")

    create_graph(conn)
    load_data(conn)
    update_graphrag(conn)
