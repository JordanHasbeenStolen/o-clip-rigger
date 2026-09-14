from core import search

TOP_K = 5

print("Type a query, or 'exit' to quit.")
while True:
    try:
        q = input("\nQuery: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not q or q.lower() == "exit":
        break
    for r, item in enumerate(search(q, TOP_K), 1):
        print(f"{r}. {item['score']:.4f}  {item['path']}")