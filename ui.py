import numpy as np
import streamlit as st
import requests as re
import apikey

def posters(ids,movie_names):
    imgs = []
    cols = st.columns(5)
    api_key = apikey.api_key

    for movieid in ids:
        response = re.get(f'https://api.themoviedb.org/3/movie/{movieid}?api_key={api_key}')
        api = response.json()

        img_path = api.get('poster_path')
        img_config = re.get(f'https://api.themoviedb.org/3/configuration?api_key={api_key}')
        img_config_json = img_config.json()

        base_url = img_config_json.get('images').get('base_url')
        img_url = base_url+'w200'+img_path

        response = re.get(img_url)
        imgs.append(response.content)

    col = np.arange(0,5,1)
    col_idx = np.resize(col,len(ids))
    idxs = np.arange(0,len(ids),1)

    for idx,img_idx in zip(col_idx,idxs):
        cols[idx].image(imgs[img_idx],use_column_width=True)
        cols[idx].markdown(f"[{movie_names[img_idx]}](https://www.themoviedb.org/movie/{ids[img_idx]})")

def main():
    st.set_page_config(layout="wide")

    ids = []
    movie_names = []
    
    st.title("Movie Recommender")
    choice = st.sidebar.selectbox('Menu', ['Home','Movies'])
    
    if choice == 'Home':

        st.header("Popular Movies")
        n = st.text_input('',placeholder='Number of movies')

        if st.button('Go'):
            response2 = re.get(f'http://127.0.0.1:8000/popular_movies?n={int(n)}')
            pop_dict = response2.json()
            for id in pop_dict.get('tmdbId').values():
                ids.append(id)
            for name in pop_dict.get('title').values():
                movie_names.append(name)
            posters(ids,movie_names)

    if choice == 'Movies':
        search = st.text_input('',placeholder='Search')

        if st.button('Search'):
            response1 = re.get(f'http://127.0.0.1:8000/recommend_movies?search={search}')
            movie_dict = response1.json()
            for id in movie_dict.get('tmdbId').values():
                ids.append(id)
            for name in movie_dict.get('title').values():
                movie_names.append(name)
            posters(ids,movie_names)

if __name__ == '__main__':
    main()