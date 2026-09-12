"""Script to launch the FastAPI server and interactive web UI."""

import uvicorn

def main():
    print("\n" + "=" * 70)
    print("  Launching Hiver AI Customer Support Agent Console (@AmazonHelp)  ")
    print("  Web Console & API: http://localhost:8000                        ")
    print("  Swagger API Docs:  http://localhost:8000/docs                   ")
    print("=" * 70 + "\n")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
