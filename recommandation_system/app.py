import streamlit as st
import joblib

df = joblib.load("data_frame.pkl")
knn = joblib.load("knn_model.pkl")

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = df.iloc[0].id


def recommend_movies(id, no_of_movies, rating_weight=0.2):
    movie_df = df.loc[df["id"] == id]

    if movie_df.empty:
        return []

    movie_row = movie_df.iloc[0]

    query_vector = movie_row["vector"]

    distances, indices = knn.kneighbors(
        [query_vector], n_neighbors=no_of_movies + 1
    )

    indices = indices[0][1:]
    distances = distances[0][1:]

    recommended_df = df.iloc[indices].copy()

    recommended_df["similarity_score"] = 1 - distances

    # Reranking Part
    rating_norm = recommended_df["rating"] / 10.0

    recommended_df["boosted_score"] = (
        recommended_df["similarity_score"]
        * (1 + rating_weight * rating_norm)
    )

    recommended_df = recommended_df.sort_values(
        by="boosted_score", ascending=False
    )

    recommended_df = recommended_df.drop(columns=["boosted_score"])

    return recommended_df


st.title("Movie Recommendation System")



# # ---- Movie Search ----

movie_options = {}

for index, row in df.iterrows():
    movie_options[row.title] = row.id

selected_index = list(movie_options.values()).index(
    st.session_state.selected_movie_id
)

selected_label = st.selectbox(
    "Search a movie",
    options=list(movie_options.keys()),
    index=selected_index
)

st.session_state.selected_movie_id = movie_options[selected_label]

st.subheader("Selected Movie")

selected_movie = df[df["id"] == st.session_state.selected_movie_id].iloc[0]

st.markdown(f"### {selected_movie.title}")
st.markdown(f"Rating: {selected_movie.rating}")
st.markdown(f"Genres: {selected_movie.genres}")
st.markdown(f"Actors: {selected_movie.actors}")
st.divider()

# # ---- Recommendations ----

recommended_df = recommend_movies(selected_movie.id, 10)
st.subheader("Recommended Movies")
cols = st.columns(2)

for idx, (_, row) in enumerate(recommended_df.iterrows()):
    with cols[idx % 2]:
        st.markdown(f"### {row.title}")
        st.markdown(f"Rating: {row.rating}")
        st.markdown(f"Genres: {row.genres}")
        st.markdown(f"Actors: {row.actors}")
        st.markdown(f"**Score:** {row.similarity_score:.2f}")

        if st.button("Select", key=row.id):
            st.session_state.selected_movie_id = row.id
            st.rerun()

        st.divider()
