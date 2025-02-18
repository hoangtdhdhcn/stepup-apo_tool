import os
import numpy as np
import time
import pickle
import openai
import prompt_utils


class LongTermMemoryManager:
    """Manages long-term memory for the assistant. It can store, retrieve, and analyze previous conversations."""

    def __init__(self, memories_folder_path, session_start_date_tuple):
        self.memories_folder_path = memories_folder_path
        self.date_start, self.day_of_week_start, self.time_start = session_start_date_tuple
        self.memories = []
        self.memories_embeddings = []
        self.memories_filepaths = []
        self.load_memories()

    def load_memories(self):
        """Loads stored memory embeddings from disk."""
        self.memories.clear()
        self.memories_embeddings.clear()
        self.memories_filepaths.clear()

        if not os.path.exists(self.memories_folder_path):
            os.makedirs(self.memories_folder_path, exist_ok=True)
            print("Memory folder not found. Created a new one.")
            return

        for entry in os.scandir(self.memories_folder_path):
            if entry.is_file() and entry.name.endswith(".pkl"):
                try:
                    with open(entry.path, "rb") as f:
                        memory = pickle.load(f)
                        self.memories.append(memory)
                        self.memories_embeddings.append(memory["embedding"])
                        self.memories_filepaths.append(entry.path)
                except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
                    print(f"Warning: Failed to load memory {entry.name} ({e})")

        self.memories_embeddings = np.array(self.memories_embeddings) if self.memories_embeddings else np.array([])
        print(f"Loaded {len(self.memories)} memories.")

    def store_conversation_seq_memory(self, conversation_sequence, reload_memories=False):
        """Stores a conversation sequence in long-term memory."""
        memory = {
            "memory_title": self.create_title_to_conversation_seq(conversation_sequence),
            "memory_string": self.convert_conversation_seq_to_string(conversation_sequence),
            "datetime": prompt_utils.get_current_time(),
            "embedding": self.get_embedding_from_conversation_seq(conversation_sequence),
            "conversation_sequence": conversation_sequence,
        }
        memory_filepath = os.path.join(self.memories_folder_path, f"{memory['memory_title']}.pkl")

        try:
            with open(memory_filepath, "wb") as f:
                pickle.dump(memory, f)
        except Exception as e:
            print(f"Error: Failed to store memory ({e})")

        if reload_memories:
            self.load_memories()

    def fetch_memory_related_to_conversation_seq(self, conversation_sequence_query, num_neighbors=3, min_similarity=0.4, minimal_output=False):
        """Finds the most relevant past conversations based on similarity."""

        if not self.memories:
            return [] if minimal_output else ([], {"memory_indices": [], "memory_similarities": [], "neighbors_filepaths": []})

        num_neighbors = min(num_neighbors, len(self.memories))
        embedding = self.get_embedding_from_conversation_seq(conversation_sequence_query)

        if self.memories_embeddings.size == 0:
            return [] if minimal_output else ([], {"memory_indices": [], "memory_similarities": [], "neighbors_filepaths": []})

        # Compute similarities
        similarities = np.dot(self.memories_embeddings, np.array(embedding)[:, np.newaxis]).flatten()
        sorted_indices = np.argsort(similarities)[::-1][:num_neighbors]

        # Filter by similarity threshold
        neighbors, memory_indices, memory_similarities, neighbors_filepaths = [], [], [], []
        for i in sorted_indices:
            if similarities[i] > min_similarity:
                neighbors.append(self.memories[i])
                memory_indices.append(i)
                memory_similarities.append(similarities[i])
                neighbors_filepaths.append(self.memories_filepaths[i])

        auxiliary_output = {
            "memory_indices": memory_indices,
            "memory_similarities": memory_similarities,
            "neighbors_filepaths": neighbors_filepaths,
        }

        return neighbors if minimal_output else (neighbors, auxiliary_output)

    def convert_conversation_seq_to_string(self, conversation_sequence):
        """Converts a conversation sequence into a structured string format."""
        date, day_of_week, time_now = prompt_utils.get_current_time()
        return f"Conversation start: ({self.date_start}, {self.day_of_week_start}, {self.time_start})\n" \
               f"Conversation end: ({date}, {day_of_week}, {time_now})\n\n" + \
               "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation_sequence)

    def get_embedding_from_conversation_seq(self, conversation_sequence):
        """Generates an embedding for a conversation sequence."""
        return self.get_embedding_from_string(self.convert_conversation_seq_to_string(conversation_sequence))

    def get_embedding_from_string(self, input_string):
        """Fetches an embedding from OpenAI API with error handling."""
        try:
            response = openai.Embedding.create(input=input_string, model="text-embedding-ada-002")
            return response["data"][0]["embedding"]
        except openai.error.OpenAIError as e:
            print(f"Error: Failed to generate embedding ({e})")
            return np.zeros(1536)  # Fallback to a zero-vector

    def create_title_to_conversation_seq(self, conversation_sequence):
        """Creates a title for the memory file based on date and message count."""
        curr_date, day_of_week, curr_time = prompt_utils.get_current_time()
        return f"mem__{curr_date.replace('/', '_')}_{day_of_week}_{curr_time[:-3]}__len_{len(conversation_sequence)}"
