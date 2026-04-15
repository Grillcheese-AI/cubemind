from cubemind.model1 import CubeMind
import numpy as np

class DocumentIngestor:
    def __init__(self, model: CubeMind, chunk_size: int = 128):
        self.model = model
        self.chunk_size = chunk_size

    def ingest(self, text: str, chunk_size: int = 200) -> tuple[list[str], list[np.ndarray]]:
        # 1. Split the text into human-readable chunks
        tokens = text.split()
        chunks = [" ".join(tokens[i:i + chunk_size]) 
                  for i in range(0, len(tokens), chunk_size)]
        
        context_vectors = []
        
        # 2. Convert each text chunk into a VSA vector
        for chunk in chunks:
            phi = self.model.encoder.encode(chunk)
            context_vectors.append(phi)
            
            # Apply Oja-Plasticity to the codebook as we read
            if hasattr(self.model, 'plastic_codebook'):
                self.model.plastic_codebook.adapt_nearest(phi)
            
        # RETURN BOTH: The strings for us, the vectors for the model
        return chunks, context_vectors

# Usage:
# ingestor = DocumentIngestor(model)
# long_history = ingestor.ingest(big_pdf_text)
# response = model.forward("Find the part about X", context=long_history)