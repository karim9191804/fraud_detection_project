"""
app.py - Point d'entrée pour Hugging Face Spaces
"""

import os
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import de l'application Gradio
from deployment.gradio_app import create_gradio_interface

# Créer et lancer l'interface
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
