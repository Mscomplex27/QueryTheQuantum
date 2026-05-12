

import pyTigerGraph as tg

conn = tg.TigerGraphConnection(
    host="https://tg-f086e6d1-e62c-44c0-a5a2-0abbb13ad10a.tg-3452941248.i.tgcloud.io",
    graphname="GraphRAG_Hackathon",
    apiToken="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSIsImlhdCI6MTc3NzgxNzMxMSwiZXhwIjoxNzc4NDIyMTE2LCJpc3MiOiJUaWdlckdyYXBoIn0.BInescyiEo5Z64jBGz_ZSQ7yhEKFcxFqCfrbt9aQFo4"
)

print("Connected successfully")
print(conn.getVertexTypes())