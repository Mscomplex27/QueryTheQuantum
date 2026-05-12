
import pyTigerGraph as tg
import os
host = os.getenv("TG_HOST")
graphname = os.getenv("TG_GRAPH")
secret = os.getenv("TG_SECRET")
# ------------------------------------------------------------

conn = tg.TigerGraphConnection(
    host=host,
    graphname=graphname,
    gsqlSecret=secret,
    tgCloud=True
)
conn.getToken()
try:
    conn.gsql(f"CREATE GRAPH {graphname}(Chunk, Entity, has_entity)")
    print(f"Graph '{graphname}' created successfully.")
except Exception as e:
    print(f"Graph might already exist: {e}")

print(f"Connection to {graphname} successful.")