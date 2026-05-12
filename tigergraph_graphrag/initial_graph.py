
import pyTigerGraph as tg

conn = tg.TigerGraphConnection(
    host="https://tg-f086e6d1-e62c-44c0-a5a2-0abbb13ad10a.tg-3452941248.i.tgcloud.io",
    graphname="GraphRAG_Hackathon",
    username="user1",
    password="Mscomplex@27",
    useCert=False
)

# This automatically fetches and manages token internally
conn.getToken(conn.createSecret())  # fallback safe initialization

print("✅ Connected using username/password (auto token mode)")
