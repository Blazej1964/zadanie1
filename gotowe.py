import streamlit as st
import openai
from PIL import Image
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid

# Zmienne
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "notes"

# Tworzenie akordeonu w pasku bocznym
with st.sidebar.expander("Wprowadź klucz API OpenAI", expanded=True):
    api_key = st.text_input("Klucz API:", type="password")

    if api_key:
        st.success("Klucz jest OK")  

# Inicjalizacja klienta Qdrant
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=":memory:")

qdrant_client = get_qdrant_client()

def assure_db_collection_exists():
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

def generate_image_description(client, uploaded_file):
    uploaded_file.seek(0)
    try:
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        file_type = uploaded_file.type.split('/')[-1]
        image_url = f"data:image/{file_type};base64,{base64_image}"

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Stwórz opis obrazka w kilku słowach, co tam widzisz?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Wystąpił błąd przy generowaniu opisu: {str(e)}"

def generate_embeddings(client, description):
    try:
        result = client.embeddings.create(
            input=[description],
            model=EMBEDDING_MODEL,
        )
        embedding = result.data[0].embedding
        return embedding
    except Exception as e:
        return f"Wystąpił błąd przy generowaniu embeddingu: {str(e)}"

def add_note_to_db(note_text, uploaded_file, client):
    vector = generate_embeddings(client, note_text)

    uploaded_file.seek(0)
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    file_type = uploaded_file.type.split('/')[-1]
    image_url = f"data:image/{file_type};base64,{base64_image}"

    # Użycie UUID jako unikalnego identyfikatora
    note_id = str(uuid.uuid4())

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=note_id,
                vector=vector,
                payload={
                    "text": note_text,
                    "image": image_url
                },
            )
        ]
    )



def list_notes_from_db(query=None):
    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
        return [
            {
                "id": note.id,  
                "text": note.payload["text"],
                "image": note.payload.get("image"),
            } for note in notes
        ]
    else: 
        query_vector = generate_embeddings(client, query)
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=10,
        )
        return [
            {
                "id": note.id,  
                "text": note.payload["text"],
                "image": note.payload.get("image"),
            } for note in notes
        ]
    
def delete_note_from_db(note_id):
    try:
        note_id_str = str(note_id)
        print(f"Próbuję usunąć notatkę o ID: {note_id_str}")

        # Sprawdzenie, czy notatka istnieje
        existing_notes = list_notes_from_db()
        existing_note_ids = [str(note["id"]) for note in existing_notes]

        if note_id_str not in existing_note_ids:
            st.warning(f"Notatka o ID {note_id_str} nie istnieje w bazie danych.")
            return

        # Użycie poprawnej metody usuwania
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=[note_id_str]
        )
        st.success(f"Notatka o ID {note_id_str} została usunięta.")

        print(f"Notatka o ID {note_id_str} została pomyślnie usunięta.") 
    except Exception as e:
        print(f"Wystąpił błąd podczas usuwania notatki o ID {note_id}: {e}")
        st.error(f"Wystąpił błąd podczas usuwania notatki: {e}")

# Główna część aplikacji
if api_key:
    client = openai.OpenAI(api_key=api_key)
    assure_db_collection_exists()

    # Dodawanie nagłówka przed selectbox
    st.sidebar.markdown("# Wybierz opcję:")  

    # Tworzenie selectboxa
    selection = st.sidebar.selectbox("Wybierz opcję:", ["Dodaj zdjęcie", "Wyszukaj notatkę", "Galeria"])

    # Resetowanie stanu sesji po zmianie zakładki
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    # Resetujemy sesję przy zmianie zakładki
    if 'selected_option' not in st.session_state or st.session_state.selected_option != selection:
        st.session_state['uploaded_files'] = []
        st.session_state['selected_option'] = selection

    if selection == "Dodaj zdjęcie":
        st.header("Dodaj Zdjęcia do galerii:")  
        st.markdown("<h5>Wczytaj zdjęcia (maks. 5)</h5>", unsafe_allow_html=True)  
        uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Utworzymy kolumny dla przycisków
            col1, col2 = st.columns(2)

            # Przycisk generujący opisy dla wszystkich zdjęć
            with col1:
                if st.button("Generuj opisy dla wszystkich zdjęć"):
                    for uploaded_file in uploaded_files:
                        description = generate_image_description(client, uploaded_file)
                        if description and "Wystąpił błąd" not in description:
                            st.session_state[f"note_text_{uploaded_file.name}"] = description
                    st.success("Opisy zostały wygenerowane dla wszystkich zdjęć.")

            # Przycisk do zapisywania wszystkich zdjęć
            with col2:
                if st.button("Zapisz wszystkie zdjęcia"):
                    for uploaded_file in uploaded_files:
                        if f"note_text_{uploaded_file.name}" in st.session_state and st.session_state[f"note_text_{uploaded_file.name}"]:
                            add_note_to_db(note_text=st.session_state[f"note_text_{uploaded_file.name}"], uploaded_file=uploaded_file, client=client)  
                    st.success("Wszystkie notatki zostały zapisane.", icon="👍")

            # Wyświetlanie zdjęć i edytowanie notatek
            for uploaded_file in uploaded_files:
                # Wyświetlanie obrazu
                image = Image.open(uploaded_file)
                st.image(image, caption='Wczytane zdjęcie', use_container_width=True)

                # Wydziel odstęp między zdjęciami
                st.markdown("---")  # Oddzielenie zdjęć

                # Edytowanie notatki
                note_key = f"note_text_{uploaded_file.name}"
                if note_key in st.session_state:
                    st.session_state[note_key] = st.text_area(f"Edytuj notatkę dla {uploaded_file.name}", value=st.session_state[note_key])


    # Obsługa zakładki "Wyszukaj notatkę"
    elif selection == "Wyszukaj notatkę":
        query = st.text_input("Wyszukaj notatkę")
        if st.button("Szukaj"):
            notes = list_notes_from_db(query)
            if notes:
                cols = st.columns(3)
                for i, note in enumerate(notes):
                    with cols[i % 3]:
                        if note["image"]:
                            st.image(note["image"], caption="Miniaturka zdjęcia", use_container_width=True)
            else:
                st.write("Brak pasujących notatek.")

    
       
    elif selection == "Galeria":
        notes = list_notes_from_db()  # Pobierz notatki
        if notes:
            cols = st.columns(3)
            for i, note in enumerate(notes):
                with cols[i % 3]:
                    if note["image"]:
                        st.image(note["image"], width=150)
                        st.write(f"Notatka ID: {note['id']}")  # Wyświetl ID dla każdej notatki
                        # Przycisk do usuwania zdjęcia
                        if st.button(f"Usuń zdjęcie ID {note['id']}"):
                            print(f"Pr attempting to delete note with ID: {note['id']}")  # Potwierdzenie prób
                            delete_note_from_db(note['id'])  # Usunięcie notatki

        else:
            st.write("Brak zapisanych zdjęć.")
   