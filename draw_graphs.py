import subprocess
for name in ["small", "medium", "large"]:
    edges = [l.strip().split(",") for l in open(f"{name}.gph") if l.strip()]
    nodes = open(f"{name}.csv").readline().strip().replace('"', '').split(",")
    with open(f"{name}.dot", "w") as f:
        f.write("digraph G {\n  rankdir=TB; node [shape=ellipse, fontname=Helvetica];\n")
        for n in nodes:
            f.write(f'  "{n}";\n')
        for a, b in edges:
            f.write(f'  "{a}" -> "{b}";\n')
        f.write("}\n")
    subprocess.run(["dot", "-Tpdf", f"{name}.dot", "-o", f"{name}_graph.pdf"], check=True)
    subprocess.run(["dot", "-Tpng", f"{name}.dot", "-o", f"{name}_graph.png"], check=True)
    print(name, len(nodes), "nodes,", len(edges), "edges")