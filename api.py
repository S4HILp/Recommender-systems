import model # This is my python notebook(model) converted to a python file.
from fastapi import FastAPI, Path, Query

app = FastAPI()

@app.get('/recommend_movies')
def recommend_movies(search):
#     search: str = Query(default=..., min_length=3),
#     n: int or None = None,
# ):
    mov = model.recommend(search)
    return mov

@app.get("/popular_movies")
def popular_movies(n: int or None = None):
    pop = model.popular(n)
    return pop
