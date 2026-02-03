import networkx as nx, json
def build_sample_graph():
    G = nx.DiGraph()
    facts = [
        {"id":"f1","text":"Elastic collisions conserve kinetic energy and momentum."},
        {"id":"f2","text":"For small timesteps, motion is approximately constant-acceleration."},
        {"id":"f3","text":"Frictionless walls cause specular reflection with restitution near 1."},
        {"id":"f4","text":"Trajectory smoothness implies bounded second temporal derivatives."},
        {"id":"f5","text":"In incompressible flows, divergence of velocity is near zero."},
    ]
    for f in facts: G.add_node(f["id"], text=f["text"], type="fact")
    for u,v in [("f1","f2"),("f2","f4"),("f3","f1"),("f5","f4")]: G.add_edge(u,v, type="supports")
    return G
def export_jsonl(G, path):
    with open(path, "w", encoding="utf-8") as f:
        for nid, data in G.nodes(data=True):
            f.write(json.dumps({"id": nid, **data}) + "\n")
